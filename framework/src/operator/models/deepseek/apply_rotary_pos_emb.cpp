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
 * \file apply_rotary_pos_emb.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor RoPEInputCast(const Tensor &input) {
    auto inputDtype = input.GetStorage()->Datatype();
    if (inputDtype == DataType::DT_FP32) { // fp32，不需要进行cast
        return input;
    }

    // RoPE: bf16->fp32
    return Cast(input, DataType::DT_FP32);
}

Tensor RotateHalf(const Tensor &input) {
    auto shape = input.GetShape();
    auto shapeSize = shape.size();
    assert(shapeSize >= 1 && "rope rotate_half input dim less than 1");
    assert(shape[shapeSize - 1] % NUM2 == 0 && "rope rotate_half last dim shape is even.");

    shape[shapeSize - 1] /= NUM2;
    std::vector<int64_t> offset1(shapeSize, 0);
    std::vector<int64_t> offset2(shapeSize, 0);
    offset2[shapeSize - 1] = shape[shapeSize - 1];

    // x1 = [..., : x.shape[-1] // 2]
    // x2 = [..., x.shape[-1] // 2 :]
    Tensor x1 = View(input, shape, offset1);
    Tensor x2 = View(input, shape, offset2);

    // cat((-x2, x1), -1)
    return Cat(
        {Mul(x2, Element(x2.GetStorage()->Datatype(), -1.0)), Add(x1, Element(x1.GetStorage()->Datatype(), 0.0))}, -1); // x1 add 0, 规避pass view+assemble未翻译registor_copy的问题
}

void ApplyRotaryPosEmbV2(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin, Tensor &qEmbed,
    Tensor &kEmbed, const int unsqueezeDim, const RoPETileShapeConfigNew &ropeTileShapeConfig) {
    auto outputDtype = qEmbed.GetStorage()->Datatype();

    // q/k仅支持四维，cos/sin仅支持san维
    assert(q.GetShape().size() == SHAPE_DIM4 && k.GetShape().size() == SHAPE_DIM4 && cos.GetShape().size() == SHAPE_DIM3 &&
           sin.GetShape().size() == SHAPE_DIM3);

    assert(!ropeTileShapeConfig.threeDimsTileShape.empty() && "rope ThreeDims Tile need to set!");
    assert(!ropeTileShapeConfig.fourDimsTileShapeQ.empty() && "rope FourDimsQ Tile need to set!");
    assert(!ropeTileShapeConfig.fourDimsTileShapeK.empty() && "rope FourDimsK Tile need to set!");
    assert(!ropeTileShapeConfig.fiveDimsTileShape.empty() && "rope FiveDims Tile need to set!");

    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeQ);              // 设置四维Tile
    auto castQ = RoPEInputCast(q);                                                        // [b,s,n,qk_d]
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeK);
    auto castK = RoPEInputCast(k);

    TileShape::Current().SetVecTile(ropeTileShapeConfig.threeDimsTileShape); // cos/sin设置san维Tile
    auto castCos = RoPEInputCast(cos);           // [b, s, qk_d]
    auto castSin = RoPEInputCast(sin);

    auto cosUnsqueeze = Unsqueeze(castCos, unsqueezeDim); // [b,1,s,qk_d]
    auto sinUnsqueeze = Unsqueeze(castSin, unsqueezeDim);

    // q=View(q, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    // q/k: [b,s,n,qk_d]
    int b = castQ.GetShape()[0];
    int s = castQ.GetShape()[1]; // use h in source code
    int h = castQ.GetShape()[2];
    int d = castQ.GetShape()[NUM_VALUE_3];

    auto qView = Reshape(castQ, {b, s, h, d / 2, 2}); // [b,n,s,qk_d//2,2]
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fiveDimsTileShape);
    auto qTrans = Transpose(qView, {NUM_VALUE_3, NUM_VALUE_4}); // [b,n,s,2,qk_d//2]
    auto qReshape = Reshape(qTrans, {b, s, h, d});              // [b,n,s,qk_d]

    // k=View(k, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    b = castK.GetShape()[0];
    s = castK.GetShape()[1];
    h = castK.GetShape()[2];
    d = castK.GetShape()[3];

    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeK);
    auto kView = Reshape(castK, {b, s, h, d / 2, 2});
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fiveDimsTileShape);
    auto kTrans = Transpose(kView, {NUM_VALUE_3, NUM_VALUE_4});
    auto kReshape = Reshape(kTrans, {b, s, h, d});

    // q_embed=(q*cos)+(rotare_half(q)*sin)
    // k_embed=(k*cos)+(rotare_half(k)*sin)
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeQ);
    qEmbed = Add(Mul(qReshape, cosUnsqueeze), Mul(RotateHalf(qReshape), sinUnsqueeze));
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeK);
    kEmbed = Add(Mul(kReshape, cosUnsqueeze), Mul(RotateHalf(kReshape), sinUnsqueeze));

    if (outputDtype != qEmbed.GetStorage()->Datatype()) {
        TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeQ);
        qEmbed = Cast(qEmbed, outputDtype);
        TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShapeK);
        kEmbed = Cast(kEmbed, outputDtype);
    }
}

void ApplyRotaryPosEmb(const Tensor &q, const Tensor &k, const Tensor &cos, const Tensor &sin,
    const Tensor &positionIds, Tensor &qEmbed, Tensor &kEmbed, const int unsqueezeDim,
    const RoPETileShapeConfig &ropeTileShapeConfig) {
    auto outputDtype = qEmbed.GetStorage()->Datatype();

    // q/k仅支持四维，cos/sin仅支持两维
    assert(q.GetShape().size() == SHAPE_DIM4 && k.GetShape().size() == SHAPE_DIM4 && cos.GetShape().size() == SHAPE_DIM2 &&
           sin.GetShape().size() == SHAPE_DIM2);

    assert(!ropeTileShapeConfig.twoDimsTileShape.empty() && "rope TwoDims Tile need to set!");
    assert(!ropeTileShapeConfig.threeDimsTileShape.empty() && "rope ThreeDims Tile need to set!");
    assert(!ropeTileShapeConfig.fourDimsTileShape.empty() && "rope FourDims Tile need to set!");
    assert(!ropeTileShapeConfig.fiveDimsTileShape.empty() && "rope FiveDims Tile need to set!");

    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShape); // 设置四维Tile
    auto castQ = RoPEInputCast(q);                                                                       // [b,n,s,qk_d]
    auto castK = RoPEInputCast(k);

    TileShape::Current().SetVecTile(ropeTileShapeConfig.twoDimsTileShape); // cos/sin设置两维Tile
    auto castCos = RoPEInputCast(cos);         // [s, qk_d]
    auto castSin = RoPEInputCast(sin);

    // cos = cos[position_ids].unsqueeze(unsqueezeDimNum)
    // sin = sin[position_ids].unsqueeze(unsqueezeDimNum)
    TileShape::Current().SetVecTile(ropeTileShapeConfig.threeDimsTileShape); // TensorIndex, 设置三维Tile
    auto cosTensorIndexes = TensorIndex(castCos, positionIds); // [s,qk_d],[b,s]->[b,s,qk_d]
    auto sinTensorIndexes = TensorIndex(castSin, positionIds);

    auto cosUnsqueeze = Unsqueeze(cosTensorIndexes, unsqueezeDim); // [b,1,s,qk_d]
    auto sinUnsqueeze = Unsqueeze(sinTensorIndexes, unsqueezeDim);

    // q=View(q, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    // q/k: [b,n,s,qk_d]
    int b = castQ.GetShape()[0];
    int h = castQ.GetShape()[1]; // use h in source code
    int s = castQ.GetShape()[2];
    int d = castQ.GetShape()[NUM_VALUE_3];

    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShape);
    auto qView = Reshape(castQ, {b, h, s, d / 2, 2}); // [b,n,s,qk_d//2,2]
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fiveDimsTileShape);
    auto qTrans = Transpose(qView, {NUM_VALUE_3, NUM_VALUE_4}); // [b,n,s,2,qk_d//2]
    auto qReshape = Reshape(qTrans, {b, h, s, d});              // [b,n,s,qk_d]

    // k=View(k, b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
    b = castK.GetShape()[0];
    h = castK.GetShape()[1];
    s = castK.GetShape()[2];
    d = castK.GetShape()[3];

    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShape);
    auto kView = Reshape(castK, {b, h, s, d / 2, 2});
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fiveDimsTileShape);
    auto kTrans = Transpose(kView, {NUM_VALUE_3, NUM_VALUE_4});
    auto kReshape = Reshape(kTrans, {b, h, s, d});

    // q_embed=(q*cos)+(rotare_half(q)*sin)
    // k_embed=(k*cos)+(rotare_half(k)*sin)
    TileShape::Current().SetVecTile(ropeTileShapeConfig.fourDimsTileShape);
    qEmbed = Add(Mul(qReshape, cosUnsqueeze), Mul(RotateHalf(qReshape), sinUnsqueeze));
    kEmbed = Add(Mul(kReshape, cosUnsqueeze), Mul(RotateHalf(kReshape), sinUnsqueeze));

    if (outputDtype != qEmbed.GetStorage()->Datatype()) {
        qEmbed = Cast(qEmbed, outputDtype);
        kEmbed = Cast(kEmbed, outputDtype);
    }
}
} // namespace npu::tile_fwk
