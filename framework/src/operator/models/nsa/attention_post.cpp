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
 * \file attention_post.cpp
 * \brief
 */
#include "attention_post.h"

#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

// b and s is dynamic, support:
// b: 16, 32, 64, 24, 48, 96
// s: 1, 2
void PostCompute(Tensor &input, PostTensors &postTensors, const PostTileConfig &tileConfig, Tensor &postOut) {
    // input: [b,s,n,kvLoraRank], fp16/bf16
    // weightUV: [n,kvLoraRank,vHeadDim], fp16/bf16/int8
    // weightUvScale: [n,1,kvLoraRank], fp32
    // weightO: [v*kvLoraRank,h], fp16/bf16/int8
    // weightOScale: [1,h], fp32
    // params check
    assert(input.GetShape().size() == SHAPE_DIM4 && postTensors.weightUV.GetShape().size() == SHAPE_DIM3 &&
            postTensors.weightO.GetShape().size() == SHAPE_DIM2);
    auto dtype = input.GetStorage()->Datatype();
    auto n = postTensors.weightUV.GetShape()[0];
    auto kvLoraRank = postTensors.weightUV.GetShape()[1];
    auto vHeadDim = postTensors.weightUV.GetShape()[2];
    auto h = postTensors.weightO.GetShape()[1];

    int tileB = tileConfig.tileB;
    int tileS = tileConfig.tileS;
    int tileBS = tileB * tileS;

    bool isQuantWUv = postTensors.weightUvScale.GetStorage() != nullptr;
    bool isSmoothWUv = postTensors.smoothScalesWUv.GetStorage() != nullptr;
    bool isQuantWo = postTensors.weightOScale.GetStorage() != nullptr;
    bool isSmoothWo = postTensors.smoothScalesWo.GetStorage() != nullptr;

    int b = input.GetShape()[0];  // SymbolicScalar b = GetInputShapeDim(input, 0);
    int s = input.GetShape()[1];  // SymbolicScalar s = GetInputShapeDim(input, 1);
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;

    LOOP("POST_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true) {
        SymbolicScalar bOffset = bIdx * tileB;
        LOOP("POST_LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, sLoop, 1)) {
            SymbolicScalar sOffset = sIdx * tileS;
            std::vector<SymbolicScalar> outOffset = {bOffset, sOffset, 0};

            TileShape::Current().SetVecTile({1, 1, 32, kvLoraRank});
            auto inputView = View(input, {tileB, tileS, n, kvLoraRank}, {bOffset, sOffset, 0, 0});
            config::SetSemanticLabel("postReshape1");
            auto inputRes = Reshape(inputView, {tileBS, n, kvLoraRank});
            TileShape::Current().SetVecTile({std::min(32, tileBS), 2, kvLoraRank});
            config::SetSemanticLabel("postTranspose1");
            auto inputTrans = Transpose(inputRes, {0, 1});  // [n,tileBS,kvLoraRank]

            config::SetSemanticLabel("postBmm");
            int c0 = 16;
            int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;
            TileShape::Current().SetCubeTile(
                {m, m}, {std::min(256L, kvLoraRank), std::min(512L, kvLoraRank)}, {vHeadDim, vHeadDim}, true);

            Tensor bmm;
            if (isQuantWUv) {
                config::SetSemanticLabel("postQuantWUv");
                TileShape::Current().SetVecTile({1, 1, std::min(512L, kvLoraRank)});
                std::tuple<Tensor, Tensor> quantRes;
                if(isSmoothWUv){
                    quantRes = Quant(inputTrans, true, true,  postTensors.smoothScalesWUv);
                }else{
                    quantRes = Quant(inputTrans, true, false);
                }
                auto inputTransQuant = std::get<0>(quantRes); // [n, tileBS, kvLoraRank], int8
                auto scaleDequant = std::get<1>(quantRes);    // [n, tileBS, 1], fp32

                // [n, tileBS, kvLoraRank] @ [n, kvLoraRank, vHeadDim] -> [n, tileBS, vHeadDim] int8 @ int8 -> int32
                auto mm = Matrix::BatchMatmul(DT_INT32, inputTransQuant,  postTensors.weightUV);

                config::SetSemanticLabel("postDequantWUv");
                TileShape::Current().SetVecTile({1, std::min(16, tileBS), std::min(32L, vHeadDim)});
                Tensor res = Cast(mm, DataType::DT_FP32);
                res = Mul(res, scaleDequant);  // [n, tileBS, VHeadDim] * [n, tileBS, 1] -> [n, tileBS, vHeadDim]
                res = Mul(res,  postTensors.weightUvScale); // [n, tileBS, VHeadDim] * [n, 1, VHeadDim] -> [n, tileBS, vHeadDim]
                bmm = Cast(res, dtype, CAST_RINT);
            } else {
                // [n,tileBS,kvLoraRank] @ [n,kvLoraRank,vHeadDim] -> [n,tileBS,vHeadDim]
                bmm = Matrix::BatchMatmul(dtype, inputTrans, postTensors.weightUV);
            }

            config::SetSemanticLabel("postTranspose2");
            TileShape::Current().SetVecTile({4, std::min(32, tileBS), vHeadDim});
            auto bmmTrans = Transpose(bmm, {0, 1}); // [n,tileBS,vHeadDim] -> [tileBS,n,vHeadDim]
            config::SetSemanticLabel("postReshape2");
            auto bmmRes = Reshape(bmmTrans, {tileBS, n * vHeadDim});

            Tensor mmRes;
            TileShape::Current().SetCubeTile({m, m}, {std::min(512L, n * vHeadDim), std::min(512L, n * vHeadDim)},
                {std::min(64L, h), std::min(64L, h)}, true);
            if (isQuantWo) {
                config::SetSemanticLabel("postQuantWo");
                TileShape::Current().SetVecTile({1, n * vHeadDim});
                std::tuple<Tensor, Tensor> quantRes;
                if (isSmoothWo) {
                    quantRes = Quant(bmmRes, true, true,  postTensors.smoothScalesWo);
                } else {
                    quantRes = Quant(bmmRes, true, false);
                }
                auto bmmResQuant = std::get<0>(quantRes);  // [tileBS, n*vHeadDim], int8
                auto scaleDequant = std::get<1>(quantRes); // [tileBS, 1], fp32

                config::SetSemanticLabel("postMm");
                // [tileBS, n*vHeadDim] @ [n*vHeadDim, h] -> [tileBS, h], int8 @ int8 -> int32
                Tensor mm = Matrix::Matmul(DataType::DT_INT32, bmmResQuant,  postTensors.weightO);

                config::SetSemanticLabel("postDequantWo");
                TileShape::Current().SetVecTile({std::min(32, tileBS), std::min(32L, h)});
                Tensor res = Cast(mm, DataType::DT_FP32);
                res = Mul(res, scaleDequant);   // [tileBS, h] * [tileBS, 1] -> [tileBS, h]
                res = Mul(res,  postTensors.weightOScale);   // [tileBS, h] * [1, h] -> [tileBS, h]
                mmRes = Cast(res, dtype, CAST_RINT);
            } else {
                // [tileBS, n*vHeadDim] @ [n*vHeadDim, h] -> [tileBS, h], dtype @ dtype -> dtype
                mmRes = Matrix::Matmul(dtype, bmmRes,  postTensors.weightO);
            }
            config::SetSemanticLabel("postReshape3");
            auto postOutView = Reshape(mmRes, {tileB, tileS, h});
            TileShape::Current().SetVecTile({1, 1, h});
            Assemble(postOutView, outOffset, postOut);
        }
    }
}

void AttentionPostStandalone(Tensor &input, PostTensors &postTensors, const PostTileConfig &tileConfig, Tensor &postOut) {
    FUNCTION("POST_MAIN",
        {input, postTensors.weightUV, postTensors.weightO, postTensors.weightUvScale, postTensors.smoothScalesWUv,
            postTensors.weightOScale, postTensors.smoothScalesWo}, {postOut}) {
        PostCompute(input, postTensors, tileConfig, postOut);
    }
}

} // namespace npu::tile_fwk
