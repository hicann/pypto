/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file statistics.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Clip(const Tensor& self, const Element& min, const Element& max)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");

    static const std::unordered_set<DataType> CLIP_A2A3_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const std::unordered_set<DataType> CLIP_A5_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(CLIP_A2A3_TYPES, CLIP_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CLIP");
    CheckTensorShapeSize(self.GetStorage(), "CLIP");

    Element min_ = min, max_ = max;

    Tensor result = self;
    if (min_.GetDataType() != DT_BOTTOM) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, min_.GetDataType() == self.GetDataType())
            << "The datatype of inputs should be same";
        result = Maximum(result, min_);
    }
    if (max_.GetDataType() != DT_BOTTOM) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, max_.GetDataType() == self.GetDataType())
            << "The datatype of inputs should be same";
        result = Minimum(result, max_);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}

Tensor Clip(const Tensor& self, const Tensor& min, const Tensor& max)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");

    static const std::unordered_set<DataType> CLIP_A2A3_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const std::unordered_set<DataType> CLIP_A5_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(CLIP_A2A3_TYPES, CLIP_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "CLIP");
    CheckTensorShapeSize(self.GetStorage(), "CLIP");

    Tensor result = self;
    if (min.GetStorage() != nullptr) {
        CheckTensorFormat(min.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");
        CheckTensorShapeSize(min.GetStorage(), "CLIP");
        CheckTensorsFormatConsistency(self.GetStorage(), min.GetStorage(), "CLIP");
        std::vector minBroadcastAxes = GetBroadcastAxes(min.GetShape(), self.GetShape());
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, minBroadcastAxes.size() <= 1)
            << "minBroadcastAxes size should be <= 1";
        CheckInt64Broadcast(self.GetStorage(), min.GetStorage(), "CLIP");
        result = Maximum(result, min);
    }
    if (max.GetStorage() != nullptr) {
        CheckTensorFormat(max.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Clip");
        CheckTensorShapeSize(max.GetStorage(), "CLIP");
        CheckTensorsFormatConsistency(self.GetStorage(), max.GetStorage(), "CLIP");
        std::vector maxBroadcastAxes = GetBroadcastAxes(max.GetShape(), self.GetShape());
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, maxBroadcastAxes.size() <= 1)
            << "maxBroadcastAxes size should be <= 1";
        CheckInt64Broadcast(self.GetStorage(), max.GetStorage(), "CLIP");
        result = Minimum(result, max);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}
// endregion: Clip

static void VarParamVaildCheck(const Tensor& input, std::vector<int>& dim)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
    CheckTensorDataType(input.GetStorage(), supportedTypes, "VAR");
    CheckTensorDimRange(input.GetStorage(), 1, NUM_VALUE_4, "VAR");
    CheckTensorShapeSize(input.GetStorage(), "VAR");

    Shape shape = input.GetShape();
    uint64_t shapeSize = shape.size();

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dim.size() <= shapeSize) << "The dim.size() should <= input.size()";
    for (uint64_t i = 0; i < shapeSize; i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape[i] > 0) << "The input shape should > 0";
    }

    if (dim.empty()) {
        for (uint64_t i = 0; i < shapeSize; i++) {
            dim.push_back(static_cast<int>(i));
        }
    }
    std::set<int> dupDimSet(dim.begin(), dim.end());

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dupDimSet.size() == dim.size()) << "There are duplicate elements in dim";
    for (size_t i = 0; i < dim.size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              dim[i] < static_cast<int>(shapeSize) && dim[i] >= -(static_cast<int>(shapeSize)))
            << "The value in dim is out of range";
        if (dim[i] < 0) {
            dim[i] = dim[i] + static_cast<int>(shapeSize);
        }
    }
    std::sort(dim.begin(), dim.end());
}

static Tensor VarResSqueeze(const Tensor& res, const std::vector<int>& dim, const std::vector<int64_t>& oriVecTile,
                            DataType dtype)
{
    std::vector<int64_t> vecTile(oriVecTile.begin(), oriVecTile.end());
    for (auto it = dim.rbegin(); it != dim.rend(); ++it) {
        vecTile.erase(vecTile.begin() + *it);
    }
    int64_t algnedSize = BLOCK_SIZE / BytesOf(dtype);
    if (vecTile.empty()) {
        vecTile.push_back(algnedSize);
    }
    int64_t lastDimSize = vecTile.back();
    if (lastDimSize % algnedSize != 0) {
        vecTile.back() = CeilDiv(lastDimSize, algnedSize) * algnedSize;
    }
    TileShape::Current().SetVecTile(vecTile);
    return Squeeze(res, dim);
}

Tensor Var(const Tensor& input, const std::vector<int>& dim, float correction, bool keepDim)
{
    CheckTensorFormat(input.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Var");

    std::vector<int> innerDim(dim.begin(), dim.end());
    VarParamVaildCheck(input, innerDim);

    DataType dtype = input.GetDataType();
    Shape shape = input.GetShape();
    auto castInput = Tensor(DT_FP32, shape);
    if (dtype == DT_FP16 || dtype == DT_BF16) {
        castInput = Cast(input, DT_FP32, CAST_NONE);
    } else {
        castInput = input;
    }

    int calcN = 1;
    auto res = castInput;
    for (size_t i = 0; i < innerDim.size(); i++) {
        calcN *= static_cast<int>(shape[innerDim[i]]);
    }
    res = Div(res, Element(DT_FP32, static_cast<float>(calcN)));
    for (size_t i = 0; i < innerDim.size(); i++) {
        res = Sum(res, innerDim[i], true);
    }

    Shape dstShape = res.GetShape();
    for (size_t i = 0; i < innerDim.size(); i++) {
        dstShape[innerDim[i]] = shape[innerDim[i]];
        res = Expand(res, dstShape);
    }

    res = Sub(castInput, res);
    res = Mul(res, res);
    float count = std::max(0.0f, static_cast<float>(calcN) - correction);
    res = Div(res, Element(DT_FP32, count));
    for (size_t i = 0; i < innerDim.size(); i++) {
        res = Sum(res, innerDim[i], true);
    }
    auto oriVecTile = TileShape::Current().GetVecTile();
    if (!keepDim) {
        res = VarResSqueeze(res, innerDim, oriVecTile.tile, dtype);
    }

    if (dtype == DT_FP16 || dtype == DT_BF16) {
        res = Cast(res, dtype, CAST_NONE);
    }
    if (!keepDim) {
        TileShape::Current().SetVecTile(oriVecTile.tile);
    }

    return res;
}

} // namespace npu::tile_fwk
