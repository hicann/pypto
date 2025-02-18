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
 * \file gen_gated_score.cpp
 * \brief
 */

#include "operator/models/deepseek/deepseek_mla.h"
#include "gen_gated_score.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

void GenGatedScoreComputePrefillPlus(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, Tensor &gatingScore) {
    DataType dType = x.GetStorage()->Datatype();
    int b = x.GetShape()[0];
    int s = x.GetShape()[1];
    int h = x.GetShape()[2];
    int n = gateW2.GetShape()[1] / 3;
    int tileB = 1;
    int L = 64;

    SymbolicScalar bLoop = b / tileB;
    
    LOOP("LOOP_L0_bIdx_gen_gated_score", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        SymbolicScalar block_num = s / L;

        LOOP("LOOP_L1_sIdx_gen_gated_score", FunctionType::DYNAMIC_LOOP, block_Idx, LoopRange(0, block_num, 1)) {
            SymbolicScalar bOfs = bIdx * tileB;
            SymbolicScalar blockStart = block_Idx * L;

            TileShape::Current().SetVecTile({tileB, 1, h});
            auto xView = View(x, {tileB, L, h}, {bOfs, blockStart, 0});
            auto xAdds = Add(xView, Element(dType, F_0));
            auto xReshape = Reshape(xAdds, {tileB * L, h});

            TileShape::Current().SetVecTile({1, h});
            TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_512, NUM_512}, {NUM_64, NUM_64});
            auto mm1Res = Matrix::Matmul(DT_FP16, xReshape, gateW1);
            auto sigmoidRes = Sigmoid(mm1Res);
            TileShape::Current().SetCubeTile({NUM_64, NUM_64}, {NUM_512, NUM_512}, {NUM_64, NUM_64});
            auto mm2Res = Matrix::Matmul(DT_FP16, sigmoidRes, gateW2);

            TileShape::Current().SetVecTile({1, n});
            auto mm2Reshape = Reshape(mm2Res, {tileB, L, 3, n});

            TileShape::Current().SetVecTile({tileB, 1, 3, n});
            auto res = Add(mm2Reshape, Element(dType, F_0));
            Assemble(res, {bOfs, blockStart, 0, 0}, gatingScore);
        }
    }
}

void GenGatedScoreComputePrefill(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, Tensor &gatingScore) {
    DataType dType = x.GetStorage()->Datatype();
    int b = x.GetShape()[0];
    int s = x.GetShape()[1];
    int h = x.GetShape()[2];
    int n = gateW2.GetShape()[1] / 3;
    int tileB = 1;
    int tileS = s;
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    LOOP("LOOP_L0_bIdx_gen_gated_score", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
        LOOP("LOOP_L1_sIdx_gen_gated_score", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            SymbolicScalar bOfs = bIdx * tileB;
            SymbolicScalar sOfs = sIdx * tileS;

            TileShape::Current().SetVecTile({tileB, 1, h});
            auto xView = View(x, {tileB, tileS, h}, {bOfs, sOfs, 0});
            auto xAdds = Add(xView, Element(dType, F_0));
            auto xReshape = Reshape(xAdds, {tileB * tileS, h});

            TileShape::Current().SetVecTile({1, h});
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_128, NUM_128});
            auto mm1Res = Matrix::Matmul(DT_FP16, xReshape, gateW1);
            auto sigmoidRes = Sigmoid(mm1Res);
            TileShape::Current().SetCubeTile({NUM_16, NUM_16}, {NUM_1024, NUM_1024}, {NUM_32, NUM_32});
            auto mm2Res = Matrix::Matmul(DT_FP16, sigmoidRes, gateW2);

            TileShape::Current().SetVecTile({1, n});
            auto mm2Reshape = Reshape(mm2Res, {tileB, tileS, 3, n});

            TileShape::Current().SetVecTile({tileB, 1, 3, n});
            auto res = Add(mm2Reshape, Element(dType, F_0));
            Assemble(res, {bOfs, sIdx, 0, 0}, gatingScore);
        }
    }
}

void GenGatedScoreFuncPrefill(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, Tensor &gatingScore) {
    FUNCTION("GENGATEDSCORE", {x, gateW1, gateW2}, {gatingScore}) {
        GenGatedScoreComputePrefillPlus(x, gateW1, gateW2, gatingScore);
    }
}

} // namespace tile_fwk