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
 * \file mla_prolog_quant_v32.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "mla_prolog_quant_v32.h"

namespace npu::tile_fwk {

static std::tuple<Tensor, Tensor> kNopeQuant(const Tensor &input) {
    constexpr float F_127 = 127.0;

    auto inputFp32 = Cast(input, DataType::DT_FP32, CAST_NONE);
    auto absRes = Abs(inputFp32);
    auto maxValue = Amax(absRes, -1, true);

    auto scaleQuant = Div(Full(Element(DT_FP32, F_127), DT_FP32, maxValue.GetShape()), maxValue);
    auto outFp32 = Mul(inputFp32, scaleQuant);
    auto outInt32 = Cast(outFp32, DataType::DT_INT32, CAST_RINT);
    auto outHalf = Cast(outInt32, DataType::DT_FP16, CAST_ROUND);
    auto outInt8 = Cast(outHalf, DataType::DT_INT8, CAST_TRUNC);

    auto scaleDeQuant = Div(Full(Element(DT_FP32, F_1), DT_FP32, scaleQuant.GetShape()), scaleQuant);
    return std::tie(outInt8, scaleDeQuant);
}

// MlaProlog Quant, b s blockNum is dynamic, support:
// b: 1, 4, 8, 16, 32, 64, 24, 48, 96, 128
// s: 1, 2
void MlaPrologQuantV32Compute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr,
    const Tensor &dequantScaleWUqQr, const Tensor &wUk, const Tensor &wDkvKr, const Tensor &rmsnormGammaCq,
    const Tensor &rmsnormGammaCkv, const Tensor &ropeCos, const Tensor &ropeSin, const Tensor &cacheIndex, Tensor &kvCache,
    Tensor &krCache, Tensor &kScaleCache, Tensor &qNormOut, Tensor &qNormScaleOut, Tensor &qNopeOut, Tensor &qRopeOut,
    Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &kScaleCacheOut, float rmsnormEpsilonCq,
    float rmsnormEpsilonCkv,const std::string& layoutKey, const MlaTileConfig &tileConfig) {
    // params check
    assert(tokenX.GetShape().size() == 3 && wUk.GetShape().size() == 3 && ropeSin.GetShape().size() == 3); // shape dim 3
    assert(kvCache.GetShape().size() == 4 && krCache.GetShape().size() == 4); // shape dim 4
    assert(layoutKey == "PA_BSND" || layoutKey == "PA_NZ");
    DataType dType = tokenX.GetDataType();
    int h = tokenX.GetShape()[2]; // 2
    // [n, qkNopeHeadDim, kvLoraRank]
    int n1 = wUk.GetShape()[0];
    int qLoraRank = wDq.GetShape()[1];
    int qkNopeHeadDim = wUk.GetShape()[1];
    int kvLoraRank = wUk.GetShape()[2];
    int qkRopeHeadDim = ropeSin.GetShape()[2]; // [b,s,qkRopeHeadDim], 2
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    // kvCache: [block_num, block_size, n2, kv_lora_rank], n2=1
    SymbolicScalar blockNum = GetInputShape(kvCache, 0);
    int blockSize = kvCache.GetShape()[1];
    int n2 = kvCache.GetShape()[2];
    assert(qkNopeHeadDim == 128 || qkRopeHeadDim == 64); // support 128, 64

    int tileBS = tileConfig.tileBS;

    RopeTileShapeConfig ropeCfg {
        {128, 128}, // 128
        {32, 128, 128}, // 32, 128
        {16, 128, 128, 128} // 16, 128
    };

    SymbolicScalar b = GetInputShape(tokenX, 0);
    SymbolicScalar s = GetInputShape(tokenX, 1);
    SymbolicScalar bsLoop  = (b * s + tileBS - 1 ) / tileBS;

    Tensor tokenX2D(tokenX.GetDataType(), {b * s, h}, "x2D");
    Tensor ropeCos2D(ropeCos.GetDataType(), {b * s, qkRopeHeadDim}, "ropeCos2D");
    Tensor ropeSin2D(ropeSin.GetDataType(), {b * s, qkRopeHeadDim}, "ropeSin2D");
    Tensor kCacheIndex2D(cacheIndex.GetDataType(), {b * s, 1}, "kCacheIndex2D");

    MlaQuantInputs quantInputs ;
    LOOP("MLA_IN_RESHAPE_LOOP", FunctionType::DYNAMIC_LOOP, unused, LoopRange(1)) {
        (void) unused;
        Reshape(tokenX,tokenX2D);
        Reshape(ropeCos, ropeCos2D);
        Reshape(ropeSin, ropeSin2D);
        Reshape(cacheIndex, kCacheIndex2D);
        if (dequantScaleWUqQr.GetStorage() != nullptr) {
            Tensor dequantScaleWUqQrReshape(dequantScaleWUqQr.GetDataType(), {1, n1 * qHeadDim});
            Reshape(dequantScaleWUqQr, dequantScaleWUqQrReshape);
            quantInputs.dequantScaleWUqQr = dequantScaleWUqQrReshape;
        }
    }

    LOOP("MLA_BS_LOOP", FunctionType::DYNAMIC_LOOP, bsIdx, LoopRange(0, bsLoop, 1)) {
        SymbolicScalar bsOffset = bsIdx * tileBS;
        std::vector<SymbolicScalar> outputOffset = {bsOffset, 0, 0};
        Tensor kNope2D, kRope2D, kScale2D;

        LOOP("MLA_PREPARE_RES", FunctionType::DYNAMIC_LOOP, unused, LoopRange(0, 1, 1)) {
            (void)unused;
            TileShape::Current().SetVecTile({tileBS, 128}); // 128
            auto tokenX2DView = View(tokenX2D, {tileBS, h}, {bsOffset, 0});
            auto qKv = PreCompute2D(tokenX2DView, wDq, wUqQr, wDkvKr, rmsnormGammaCq, rmsnormEpsilonCq, quantInputs);
            Tensor q = qKv[0];      // [b*s, n*qHeadDim]
            Tensor kvTmp = qKv[1];  // [b*s, kvLoraRank+qkRopeHeadDim]

            /******** qNorm ********/
            config::SetSemanticLabel("Assemble_qNorm");
            Tensor qNorm = qKv[2];
            TileShape::Current().SetVecTile({tileBS, qLoraRank});
            Assemble(qNorm, {bsOffset, 0}, qNormOut);  // output
            Tensor qNormScale = qKv[3];
            TileShape::Current().SetVecTile({tileBS, 1});  // 32, 64
            Assemble(qNormScale, {bsOffset, 0}, qNormScaleOut);  // output

            /******** q ********/
            auto qTmp = Reshape(q, {tileBS, n1, qHeadDim});
            config::SetSemanticLabel("Prepare_qNope");
            Tensor qNope = View(qTmp, {tileBS, n1, qkNopeHeadDim}, {0, 0, 0});  // [b,s,n,qkNopeHeadDim]
            std::vector<int64_t> tileShape = {std::min(32, tileBS), 32, qkNopeHeadDim};  // 32
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeTrans = Transpose(qNope, {0, 1});  // [n,bs,qkNopeHeadDim]

            int c0 = 16;                                        // 16
            int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;  // 32
            config::SetSemanticLabel("Matmul_qNope_wUk");
            TileShape::Current().SetCubeTile({m, m}, {128, 128}, {128, 128});  // 128
            // bmm: (n,bs,qkNopeHeadDim) @ (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
            Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

            tileShape = {1, std::min(32, tileBS), kvLoraRank};  // 32
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1});  // [bs,n,kvLoraRank]
            config::SetSemanticLabel("Assemble_queryOut");
            TileShape::Current().SetVecTile({1, 32, 128});  // 32, 128
            Assemble(qNopeNewTrans, outputOffset, qNopeOut);   // output1

            Tensor qPeView = View(qTmp, {tileBS, n1, qkRopeHeadDim}, {0, 0, qkNopeHeadDim});
            Tensor ropeCosView = View(ropeCos2D, {tileBS, qkRopeHeadDim}, {bsOffset, 0});
            Tensor ropeSinView = View(ropeSin2D, {tileBS, qkRopeHeadDim}, {bsOffset, 0});
            auto qRopeView = Rope3DV2(qPeView, ropeCosView, ropeSinView, ropeCfg);
            config::SetSemanticLabel("Assemble_qRope");
            TileShape::Current().SetVecTile({1, 32, 64});  // 32, 64
            Assemble(qRopeView, outputOffset, qRopeOut);  // output2

            /******** RoPE ********/
            TileShape::Current().SetVecTile({2, 512}); // 2, 512
            config::SetSemanticLabel("RotaryPosEmb");
            Tensor kPeView = View(kvTmp, {tileBS, qkRopeHeadDim}, {0, kvLoraRank}); // [b*s,qkRopeHeadDim]
            kRope2D = RopeV2(kPeView, ropeCosView, ropeSinView, ropeCfg);

            /******** kNope ********/
            Tensor compressedKv = View(kvTmp, {tileBS, kvLoraRank}, {0, 0}); // [b*s,kvLoraRank]
            tileShape = {2, 512}; // 2, 512
            config::SetSemanticLabel("RmsNorm_compressedKv");
            TileShape::Current().SetVecTile(tileShape);
            Tensor kNope = RmsNorm(compressedKv, rmsnormGammaCkv, rmsnormEpsilonCkv); // [b*s,kvLoraRank]
            /******** kNope Quant ********/
            // no smooth
            config::SetSemanticLabel("Quant_kNope");
            TileShape::Current().SetVecTile(32, kvLoraRank); // 32
            Tensor kNopeSplit = Reshape(kNope, {tileBS, 4, kvLoraRank / 4}); //4
            TileShape::Current().SetVecTile(32, 4, kvLoraRank / 4); // 32, 4
            auto kNopeQuantRes = kNopeQuant(kNopeSplit);
            Tensor kNopeQuant = std::get<0>(kNopeQuantRes);
            Tensor kNopeScale = std::get<1>(kNopeQuantRes);
            TileShape::Current().SetVecTile(32, 4, kvLoraRank / 4); // 32, 4
            kNope2D = Reshape(kNopeQuant, {tileBS, kvLoraRank});
            kScale2D = Reshape(kNopeScale, {tileBS, 4}); // 4
        }

        Tensor krCache2D(krCache.GetDataType(), {blockNum * blockSize * n2, qkRopeHeadDim});
        Tensor kvCache2D(kvCache.GetDataType(), {blockNum * blockSize * n2, kvLoraRank});
        Tensor kScaleCache2D(kScaleCache.GetDataType(), {blockNum * blockSize * n2, 4}); // 4
        LOOP("MLA_CACHE_RESHAPE_4D_2D", FunctionType::DYNAMIC_LOOP, unused, LoopRange(0, 1, 1)) {
            (void)unused;
            Reshape(krCache, krCache2D);
            Reshape(kvCache, kvCache2D);
            Reshape(kScaleCache, kScaleCache2D);
        }

        Tensor krCacheOut2D, kvCacheOut2D, kScaleCacheOut2D;
        LOOP("MLA_UPDATE_CACHE", FunctionType::DYNAMIC_LOOP, unused, LoopRange(0, 1, 1)) {
            (void)unused;
            auto index = View(kCacheIndex2D, {tileBS, 1}, {bsOffset, 0});
            // krCache: [blockNum, blockSize, n2, qkRopeHeadDim]
            config::SetSemanticLabel("ScatterUpdate_krCache");
            TileShape::Current().SetVecTile(32, qkRopeHeadDim); // 32
            krCacheOut2D = ScatterUpdate(krCache2D, index, kRope2D, -2, layoutKey, blockSize); // -2
            // kvCache: [blockNum, blockSize, n2, kvLoraRank]
            config::SetSemanticLabel("ScatterUpdate_kvCache");
            TileShape::Current().SetVecTile(32, kvLoraRank); // 32
            kvCacheOut2D = ScatterUpdate(kvCache2D, index, kNope2D, -2, layoutKey, blockSize); // -2
            // kScaleCache: [blockNum, blockSize, n2, 4]
            config::SetSemanticLabel("ScatterUpdate_kScaleCache");
            TileShape::Current().SetVecTile(32, 4); // 32, 4
            kScaleCacheOut2D = ScatterUpdate(kScaleCache2D, index, kScale2D, -2, layoutKey, 4); // -2, 4
        }

        LOOP("MLA_CACHE_RESHAPE_2D_4D", FunctionType::DYNAMIC_LOOP, unused, LoopRange(0, 1, 1)) {
            (void)unused;
            Reshape(krCacheOut2D, krCacheOut);
            Reshape(kvCacheOut2D, kvCacheOut);
            Reshape(kScaleCacheOut2D, kScaleCacheOut);
        }
    }
}

void MlaPrologQuantV32(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr,
    const Tensor &dequantScaleWUqQr, const Tensor &wUk, const Tensor &wDkvKr, const Tensor &rmsnormGammaCq,
    const Tensor &rmsnormGammaCkv, const Tensor &ropeCos, const Tensor &ropeSin, const Tensor &cacheIndex, Tensor &kvCache,
    Tensor &krCache, Tensor &kScaleCache, Tensor &qNormOut, Tensor &qNormScaleOut,  Tensor &qNopeOut, Tensor &qRopeOut,
    Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &kScaleCacheOut, float rmsnormEpsilonCq, float rmsnormEpsilonCkv,
    const std::string &layoutKey,const MlaTileConfig &tileConfig) {
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_MODE, NUM_4);
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{3, 4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, NUM_2 * NUM_1024 * NUM_1024);

    FUNCTION("main",
        {tokenX, wDq, wUqQr, dequantScaleWUqQr, wUk, wDkvKr, rmsnormGammaCq, rmsnormGammaCkv,  ropeCos, ropeSin, cacheIndex, kvCache, krCache, kScaleCache},
        {qNormOut, qNormScaleOut, qNopeOut, qRopeOut},
        {{kvCacheOut, kvCache}, {krCacheOut, krCache}, {kScaleCacheOut, kScaleCache}}) {
        // compute
        MlaPrologQuantV32Compute(tokenX, wDq, wUqQr, dequantScaleWUqQr, wUk, wDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeCos, ropeSin, cacheIndex,
            kvCache, krCache, kScaleCache, qNormOut, qNormScaleOut, qNopeOut, qRopeOut, kvCacheOut,
            krCacheOut, kScaleCacheOut, rmsnormEpsilonCq, rmsnormEpsilonCkv, layoutKey, tileConfig);
    }
}

} // namespace npu::tile_fwk