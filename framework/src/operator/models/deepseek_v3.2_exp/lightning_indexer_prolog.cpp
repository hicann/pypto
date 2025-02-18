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
 * \file lightning_indexer_prolog.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "lightning_indexer_prolog.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

Tensor LayerNorm(const Tensor &x, const Tensor &weight, const Tensor &bias, const int dim) {
    ASSERT(dim == (int)(x.GetShape().size() - 1) || dim == -1) << "We only support LayerNorm for the last dimension";
    ASSERT(x.GetStorage()->Datatype() == DT_FP32);
    constexpr float epsilon = 1e-6f;
    int actualDim = dim < 0 ? dim + x.GetShape().size() : dim;

    // do division first to avoid overflow
    auto xScaled = Mul(x, Element(DataType::DT_FP32, 1.0f / x.GetShape()[actualDim]));
    auto mean = Sum(xScaled, -1, true);

    auto diff = Sub(x, mean);
    auto squaredDiff = Mul(diff, diff);
    auto squaredDiffScaled = Mul(squaredDiff, Element(DataType::DT_FP32, 1.0f / x.GetShape()[actualDim]));
    auto var = Sum(squaredDiffScaled, -1, true);
    // add epsilon to avoid division by zero
    auto varEps = Add(var, Element(DT_FP32, epsilon));
    auto stdVar = Sqrt(varEps);
    auto res32 = Div(diff, stdVar);

    auto weight32 = Cast(weight, DT_FP32);
    auto bias32 = Cast(bias, DT_FP32);

    return Add(Mul(res32, weight32), bias32);
}

Tensor RotateHalfValidShape(const Tensor &input) {
    auto shape = input.GetShape();
    auto shapeSize = shape.size();
    ASSERT(shapeSize >= 1) << "rope rotate_half input dim less than 1";
    ASSERT(shape[shapeSize - 1] % NUM2 == 0) << "rope rotate_half last dim shape is even.";

    shape[shapeSize - 1] /= NUM2;
    std::vector<SymbolicScalar> offset1(shapeSize, 0);
    std::vector<SymbolicScalar> offset2(shapeSize, 0);
    offset2[shapeSize - 1] = shape[shapeSize - 1];

    std::vector<SymbolicScalar> validShape = input.GetStorage()->GetDynValidShape();
    validShape[shapeSize - 1] = validShape[shapeSize - 1] / NUM2;

    Tensor x1 = View(input, shape, validShape, offset1);
    Tensor x2 = View(input, shape, validShape, offset2);

    // cat((-x2, x1), -1)
    return Cat(
        {Mul(x2, Element(x2.GetDataType(), -1.0)), Add(x1, Element(x1.GetDataType(), 0.0))}, -1);
}

Tensor Rope3D(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM3 && cos.GetShape().size() == SHAPE_DIM2 &&
           sin.GetShape().size() == SHAPE_DIM2);

    TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
        castX = Add(castX, Element(DT_FP32, 0.0f));
    }
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);
    castCos = Reshape(castCos, {x.GetShape()[NUM_VALUE_0], 1, x.GetShape()[NUM_VALUE_2]});
    castSin = Reshape(castSin, {x.GetShape()[NUM_VALUE_0], 1, x.GetShape()[NUM_VALUE_2]});

    std::vector<SymbolicScalar> xValidShape = x.GetStorage()->GetDynValidShape();
    auto xView = Reshape(castX,
        {x.GetShape()[NUM_VALUE_0], x.GetShape()[NUM_VALUE_1], x.GetShape()[NUM_VALUE_2] / NUM_VALUE_2, NUM_VALUE_2},
        {xValidShape[NUM_VALUE_0], xValidShape[NUM_VALUE_1], xValidShape[NUM_VALUE_2] / NUM_VALUE_2, NUM_VALUE_2});
    TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128, NUM_128);
    auto xTrans = Transpose(xView, {NUM_VALUE_2, NUM_VALUE_3});
    auto xReSecond = Reshape(xTrans, x.GetShape(), xValidShape);

    TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128, NUM_128);
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalfValidShape(xReSecond), castSin));
    auto res = Cast(xEmbed, x.GetStorage()->Datatype());
    return res;
}

Tensor Rope(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM2 && cos.GetShape().size() == SHAPE_DIM2 &&
           sin.GetShape().size() == SHAPE_DIM2);

    auto seqSize = x.GetShape()[NUM_VALUE_0];
    auto dR = x.GetShape()[NUM_VALUE_1];
    auto xDtype = x.GetStorage()->Datatype();

    TileShape::Current().SetVecTile(tileConfig.twoDim[NUM_VALUE_0], tileConfig.twoDim[NUM_VALUE_1]);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
        castX = Add(castX, Element(DT_FP32, 0.0f));
    }
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);

    auto xView = Reshape(castX, {1, seqSize, dR / NUM_VALUE_2, NUM_VALUE_2});
    TileShape::Current().SetVecTile(tileConfig.fourDim[NUM_VALUE_0], tileConfig.fourDim[NUM_VALUE_1],
        tileConfig.fourDim[NUM_VALUE_2], tileConfig.fourDim[NUM_VALUE_3]);
    auto xTrans = Transpose(xView, {NUM_VALUE_2, NUM_VALUE_3});
    auto xReSecond = Reshape(xTrans, {seqSize, dR});

    TileShape::Current().SetVecTile(tileConfig.twoDim[NUM_VALUE_0], tileConfig.twoDim[NUM_VALUE_1]);

    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin));
    auto res = Cast(xEmbed, xDtype);
    return res;
}

void LightningIndexerPrologCompute(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params) {
    SymbolicScalar b = GetInputShape(inputs.x, 0);
    SymbolicScalar seq = GetInputShape(inputs.x, 1);
    int headDim = params.headDim;
    int ropeHeadDim = params.ropeHeadDim;
    int qLoraRank = params.qLoraRank;
    int dim = params.dim;
    int headNum = params.headNum;

    Tensor x2D(inputs.x.GetStorage()->Datatype(), {b * seq, dim}, "x2D");
    Tensor qr2D(inputs.qr.GetStorage()->Datatype(), {b * seq, qLoraRank}, "qr2D");
    Tensor cos2D(inputs.cos.GetStorage()->Datatype(), {b * seq, ropeHeadDim}, "cos2D");
    Tensor sin2D(inputs.sin.GetStorage()->Datatype(), {b * seq, ropeHeadDim}, "sin2D");
    Tensor lnW2D(inputs.lnW.GetStorage()->Datatype(), {1, inputs.lnW.GetShape()[0]});
    Tensor lnBias2D(inputs.lnBias.GetStorage()->Datatype(), {1, inputs.lnBias.GetShape()[0]});

    LOOP("LOOP_RESHAPE_IN", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
        (void) batchId;
        x2D = Reshape(inputs.x, {b * seq, dim}, true);
        qr2D = Reshape(inputs.qr, {b * seq, qLoraRank}, true);
        cos2D = Reshape(inputs.cos, {b * seq, ropeHeadDim}, true);
        sin2D = Reshape(inputs.sin, {b * seq, ropeHeadDim}, true);
        lnW2D = Reshape(inputs.lnW, {1, inputs.lnW.GetShape()[0]}, true);
        lnBias2D = Reshape(inputs.lnBias, {1, inputs.lnBias.GetShape()[0]}, true);
    }

    std::set<int> unrollList = {1, 2, 4, 8, 16, 32};
    LOOP("IndexerPrologLoop", FunctionType::DYNAMIC_LOOP, bsIdx, LoopRange(b * seq), unrollList) {
        for (int unrollLength : unrollList) {
            UNROLL(unrollLength) {
                int tileBS = unrollLength;
                SymbolicScalar actBS = tileBS;
                auto c1Tile = params.indexerTileConfigs.c1TileShape;
                auto v1Tile = params.indexerTileConfigs.v1TileShape;

                config::SetSemanticLabel("QMatmul");
                TileShape::Current().SetCubeTile({c1Tile[NUM_VALUE_0], c1Tile[NUM_VALUE_1]},
                    {c1Tile[NUM_VALUE_2], c1Tile[NUM_VALUE_3]}, {c1Tile[NUM_VALUE_4], c1Tile[NUM_VALUE_5]}, true);
                auto qrBlock = View(qr2D, {tileBS, qLoraRank}, {actBS, qLoraRank}, {bsIdx, 0});
                // {tileBS, qLoraRank} * {qLoraRank, headNum * headDim} = {tileBS, headNum * headDim}
                auto q32 = Matrix::Matmul<false, false>(DT_FP32, qrBlock, inputs.qW);

                config::SetSemanticLabel("QCast");
                TileShape::Current().SetVecTile(std::min(tileBS, NUM_4), NUM_32, v1Tile[NUM_VALUE_1]);
                auto q = Cast(Reshape(q32, {tileBS, headNum, headDim}), qrBlock.GetStorage()->Datatype());
                Tensor qRope = View(q, {tileBS, headNum, ropeHeadDim}, {actBS, headNum, ropeHeadDim}, {0, 0, 0});
                Tensor qNope = View(q, {tileBS, headNum, headDim - ropeHeadDim},
                    {actBS, headNum, headDim - ropeHeadDim}, {0, 0, ropeHeadDim});
                qNope = Cast(Cast(qNope, DT_FP32), qNope.GetDataType());

                config::SetSemanticLabel("KMatmul");
                auto c2Tile = params.indexerTileConfigs.c2TileShape;
                TileShape::Current().SetCubeTile({c2Tile[NUM_VALUE_0], c2Tile[NUM_VALUE_1]},
                    {c2Tile[NUM_VALUE_2], c2Tile[NUM_VALUE_3]}, {c2Tile[NUM_VALUE_4], c2Tile[NUM_VALUE_5]}, true);
                TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1], v1Tile[NUM_VALUE_1]);
                auto xBlock = View(x2D, {tileBS, dim}, {actBS, dim}, {bsIdx, 0});
                // {tileBS, dim} * {dim, headNum} = {tileBS, headNum}
                auto weights = Matrix::Matmul<false, false>(inputs.x.GetStorage()->Datatype(), xBlock, inputs.projW);
                Assemble(weights, {bsIdx, 0}, outputs.weight);

                // {tileBS, dim} * {dim, headDim} = {tileBS, headDim}
                auto k = Matrix::Matmul<false, false>(DT_FP32, xBlock, inputs.kW);
                k = Cast(LayerNorm(k, lnW2D, lnBias2D, -1), xBlock.GetStorage()->Datatype()); // {tileBS, headDim}
                Tensor kRope = View(k, {tileBS, ropeHeadDim}, {actBS, ropeHeadDim}, {0, 0});
                Tensor kNope =
                    View(k, {tileBS, headDim - ropeHeadDim}, {actBS, headDim - ropeHeadDim}, {0, ropeHeadDim});

                TileShape::Current().SetVecTile(
                    v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1], v1Tile[NUM_VALUE_2]);
                auto cos2DView = View(cos2D, {tileBS, ropeHeadDim}, {actBS, ropeHeadDim}, {bsIdx, 0});
                auto sin2DView = View(sin2D, {tileBS, ropeHeadDim}, {actBS, ropeHeadDim}, {bsIdx, 0});

                config::SetSemanticLabel("QRope");
                // qRope{tileBS * headNum, ropeHeadDim}  cos{tileBS, ropeHeadDim}   sin{tileBS, ropeHeadDim}
                config::SetSemanticLabel("KRope");
                auto qRoped = Rope3D(qRope, cos2DView, sin2DView, params.ropeTileConfigs); // {tileBS, headNum, ropeHeadDim}
                TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1]);
                // kRope{tileBS, ropeHeadDim}  cos{tileBS, ropeHeadDim}   sin{tileBS, ropeHeadDim}
                auto kRoped = Rope(kRope, cos2DView, sin2DView, params.ropeTileConfigs); // {tileBS, ropeHeadDim}

                config::SetSemanticLabel("KAssemble");
                TileShape::Current().SetVecTile(NUM_1, NUM_32, NUM_128, NUM_128);
                Assemble({{qRoped, {bsIdx, 0, 0}}, {qNope, {bsIdx, 0, ropeHeadDim}}}, outputs.query, true);

                TileShape::Current().SetVecTile(tileBS, NUM_256);
                auto kType = kNope.GetDataType();
                kNope = Cast(Cast(kNope, DT_FP32), kType);
                auto kUpdate = Cat({kRoped, kNope}, -1); // {tileBS, headDim}
                auto kUpdate4D = Reshape(kUpdate, {tileBS, 1, 1, headDim});
                auto index = View(inputs.kCacheIndex, {tileBS, 1}, {actBS, 1}, {bsIdx, 0});

                TileShape::Current().SetVecTile(tileBS, NUM_128, NUM_128, NUM_128);
                outputs.kCacheOut =
                    ScatterUpdate(inputs.kCache, index, kUpdate4D, SCATTER_UPADATE_DIM, "PA_BSND", params.blockSize);
            }
        }
    }
}

void LightningIndexerProlog(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params) {
    FUNCTION("LightningIndexerProlog",
        {
            inputs.x, inputs.qr, inputs.qW, inputs.kW, inputs.projW, inputs.lnW, inputs.lnBias, inputs.cos, inputs.sin,
            inputs.kCache, inputs.kCacheIndex, inputs.blockTable
    },
        {outputs.query, outputs.weight}, {{outputs.kCacheOut, inputs.kCache}}) {
        LightningIndexerPrologCompute(inputs, outputs, params);
    }
}

} // namespace npu::tile_fwk