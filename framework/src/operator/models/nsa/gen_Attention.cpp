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
 * \file gen_Attention.cpp
 * \brief
 */
 
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "gen_Attention.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void GenAttentionCompute(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut, GenAttenTileShapeConfig &tileConfig) {
    int nDimSize = cmpAtten.GetShape()[2];
    int dDimSize = cmpAtten.GetShape()[3];
    int dGateDimSize = gatingScore.GetShape()[3];
    int tileB = tileConfig.tileBSize;
    int tileS = tileConfig.tileS1Size;
    auto v1Tile = tileConfig.vec1TileShape;
    auto v2Tile = tileConfig.vec2TileShape;

    SymbolicScalar bDimSize = GetInputShape(cmpAtten, 0);
    SymbolicScalar sDimSize = GetInputShape(cmpAtten, 1);
    SymbolicScalar bLoop = bDimSize / tileB;
    SymbolicScalar sLoop = sDimSize / tileS;
    DataType dType = cmpAtten.GetStorage()->Datatype();
    LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(bLoop)) {
        SymbolicScalar bOffset = bIdx * tileB;
        SymbolicScalar actualBSize = std::min(tileB, (bDimSize - bIdx * tileB));
        LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(sLoop)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0, 0};
            SymbolicScalar actualsSize = std::min(tileS, (sDimSize - sIdx * tileS));
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1], v1Tile[2], v1Tile[3]);
            auto cmpAttenTile = View(cmpAtten, {tileB, tileS, nDimSize, dDimSize},
                {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
            auto selAttenTile = View(selAtten, {tileB, tileS, nDimSize, dDimSize},
                {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
            auto winAttenTile = View(winAtten, {tileB, tileS, nDimSize, dDimSize},
                {actualBSize, actualsSize, nDimSize, dDimSize}, {bOffset, sOffset, 0, 0});
            auto cmpAttenFP32Tile = Cast(cmpAttenTile, DT_FP32);
            auto selAttenFP32Tile = Cast(selAttenTile, DT_FP32);
            auto winAttenFP32Tile = Cast(winAttenTile, DT_FP32);
            TileShape::Current().SetVecTile(v2Tile[0], v2Tile[1], v2Tile[2], v2Tile[3]);
            auto gatingScoreTile = View(gatingScore, {tileB, tileS, nDimSize, dGateDimSize},
                {actualBSize, actualsSize, nDimSize, dGateDimSize}, {bOffset, sOffset, 0, 0});
            auto gatingScoreFP32 = Cast(gatingScoreTile, DT_FP32);
            auto cmpWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 0});
            auto selWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 1});
            auto winWeight = View(gatingScoreFP32, {tileB, tileS, nDimSize, 1}, {0, 0, 0, 2});
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1], v1Tile[2], v1Tile[3]);
            auto mulCmp = Mul(cmpAttenFP32Tile, cmpWeight);
            auto mulSel = Mul(selAttenFP32Tile, selWeight);
            auto mulWin = Mul(winAttenFP32Tile, winWeight);
            auto addCmpSel = Add(mulCmp, mulSel);
            auto outFP32 = Add(addCmpSel, mulWin);
            TileShape::Current().SetVecTile(v1Tile[0], v1Tile[1], v1Tile[2], v1Tile[3]);
            auto attentionOutTile = Cast(outFP32, dType, CAST_RINT);
            Assemble(attentionOutTile, outOffset, attentionOut);
        }
    }
}

void GenAttention(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut, GenAttenTileShapeConfig &tileConfig) {
    FUNCTION("main", {cmpAtten, selAtten, winAtten, gatingScore}, {attentionOut}) {
        GenAttentionCompute(cmpAtten, selAtten, winAtten, gatingScore, attentionOut, tileConfig);
}
}

} // namespace tile_fwk
