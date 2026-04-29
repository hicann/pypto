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
* \file quantization.cpp
* \brief Quantization operation implementation for INT8 symmetric and asymmetric quantization
*/

#include "interface/utils/operator_tracer.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/vector/tensor_transformation.h"

#include <unordered_set>

namespace npu::tile_fwk {

namespace {
constexpr size_t QUANT_MX_MIN_RANK = 1;
constexpr size_t QUANT_MX_MAX_RANK = 4;
constexpr int64_t QUANT_MX_GROUP_COLS = 32;
constexpr int64_t QUANT_MX_SCALE_GROUP_COLS = 64;
constexpr int64_t QUANT_MX_SCALE_PAIR_SIZE = 2;
constexpr int64_t QUANT_MX_TILE_ALIGN_BYTES = 256;
const std::unordered_set<DataType> QUANT_MX_SUPPORTED_INPUT_TYPES = {DataType::DT_FP16, DataType::DT_BF16,
                                                                     DataType::DT_FP32};
const std::unordered_set<DataType> QUANT_MX_SUPPORTED_OUTPUT_TYPES = {DataType::DT_FP8E4M3};

int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, divisor != 0) << "CeilDiv divisor must not be zero.";
    return (dividend + divisor - 1) / divisor;
}

void CheckQuantMXDtype(DataType quantDtype)
{
    ASSERT(
        VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
        QUANT_MX_SUPPORTED_OUTPUT_TYPES.find(quantDtype) != QUANT_MX_SUPPORTED_OUTPUT_TYPES.end())
        << "QuantMX currently only supports DT_FP8E4M3 output. Current quant dtype: " << DataType2String(quantDtype);
}

void CheckQuantMXMode(DequantScaleRoundingMode mode)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, mode == DequantScaleRoundingMode::ROUND_DOWN)
        << "QuantMX currently only supports ROUND_DOWN (OCP standard) mode.";
}

void CheckQuantMXPerformanceMode(int64_t performanceMode)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, performanceMode != 0)
        << "QuantMX currently only supports performance mode.";
}

DequantScaleRoundingMode GetQuantMXMode(const Operation& op)
{
    int64_t modeValue = 0;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantMode, modeValue))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantMode;
    return static_cast<DequantScaleRoundingMode>(modeValue);
}

int64_t GetQuantMXAxis(const Operation& op)
{
    int64_t axis = 0;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantAxis, axis))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantAxis;
    return axis;
}

int64_t GetQuantMXPerformanceMode(const Operation& op)
{
    int64_t performanceMode = 0;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantPerformanceMode, performanceMode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantPerformanceMode;
    return performanceMode;
}

int64_t NormalizeQuantMXAxis(int64_t axis, size_t rank)
{
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < static_cast<int64_t>(rank))
        << "QuantMX axis is out of range. Current axis: " << axis << ", input rank: " << rank;
    return axis;
}

void CheckQuantMXAxis(int64_t axis, size_t rank)
{
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, rank);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == static_cast<int64_t>(rank) - 1)
        << "QuantMX currently only supports the last axis. Current axis: " << axis << ", input rank: " << rank;
}

void CheckQuantMXInput(const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis)
{
    const auto inputDtype = input.GetDataType();
    ASSERT(
        VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
        QUANT_MX_SUPPORTED_INPUT_TYPES.find(inputDtype) != QUANT_MX_SUPPORTED_INPUT_TYPES.end())
        << "QuantMX currently only supports DT_FP16, DT_BF16, and DT_FP32 input.";
    CheckQuantMXDtype(quantDtype);
    CheckQuantMXMode(mode);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantMX only supports TILEOP_ND input.";
    ASSERT(
        VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
        QUANT_MX_MIN_RANK <= input.GetShape().size() && input.GetShape().size() <= QUANT_MX_MAX_RANK)
        << "QuantMX only supports 1D to 4D input.";
    CheckQuantMXAxis(axis, input.GetShape().size());
    const int64_t lastDimBytes = input.GetShape().back() * BytesOf(inputDtype);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX view shape's last dim must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;
}

void CheckQuantMXPerformanceTileShape(const LogicalTensorPtr& input, const VecTile& vecTile, int64_t performanceMode)
{
    if (performanceMode == 0) {
        return;
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX performance mode tile shape rank must match input rank.";
    const int64_t lastTileDim = vecTile[vecTile.size() - 1];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastTileDim == input->GetShape().back())
        << "QuantMX performance mode requires tile shape last dim to be the same as input last dim. Current tile "
           "last dim: "
        << lastTileDim << ", input last dim: " << input->GetShape().back();
}

std::vector<int64_t> BuildQuantMXGroupedShape(const std::vector<int64_t>& inputShape)
{
    auto groupedShape = inputShape;
    groupedShape.back() = CeilDiv(groupedShape.back(), QUANT_MX_GROUP_COLS);
    return groupedShape;
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedShape(const std::vector<int64_t>& inputShape)
{
    if (inputShape.size() == 1) {
        return {CeilDiv(inputShape[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedShape;
    groupedShape.reserve(inputShape.size() - 1);
    for (size_t i = 0; i + 2 < inputShape.size(); ++i) {
        groupedShape.push_back(inputShape[i]);
    }
    groupedShape.push_back(inputShape[inputShape.size() - 2] * CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS));
    return groupedShape;
}

std::vector<int64_t> BuildQuantMXPerformanceVecTile(const std::vector<int64_t>& inputVecTile)
{
    if (inputVecTile.size() == 1) {
        return {CeilDiv(inputVecTile[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedVecTile;
    groupedVecTile.reserve(inputVecTile.size() - 1);
    for (size_t i = 0; i + 2 < inputVecTile.size(); ++i) {
        groupedVecTile.push_back(inputVecTile[i]);
    }
    groupedVecTile.push_back(
        inputVecTile[inputVecTile.size() - 2] * CeilDiv(inputVecTile.back(), QUANT_MX_GROUP_COLS));
    return groupedVecTile;
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedOffset(
    const std::vector<int64_t>& inputOffset, const std::vector<int64_t>& inputShape,
    const std::vector<int64_t>& inputTileShape)
{
    if (inputOffset.size() == 1) {
        return {inputOffset[0] / QUANT_MX_GROUP_COLS};
    }

    std::vector<int64_t> groupedOffset;
    groupedOffset.reserve(inputOffset.size() - 1);
    for (size_t i = 0; i + 2 < inputOffset.size(); ++i) {
        groupedOffset.push_back(inputOffset[i]);
    }
    const int64_t groupCols = CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS);
    const int64_t tileRows = inputTileShape[inputTileShape.size() - 2];
    groupedOffset.push_back(
        inputOffset[inputOffset.size() - 2] * groupCols + tileRows * (inputOffset.back() / QUANT_MX_GROUP_COLS));
    return groupedOffset;
}

std::vector<SymbolicScalar> BuildQuantMXGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto groupedValidShape = inputValidShape;
    groupedValidShape.back() = (groupedValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS;
    return groupedValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXPerformanceGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    if (inputValidShape.size() == 1) {
        return {(inputValidShape[0] + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS};
    }

    std::vector<SymbolicScalar> groupedValidShape;
    groupedValidShape.reserve(inputValidShape.size() - 1);
    for (size_t i = 0; i + 2 < inputValidShape.size(); ++i) {
        groupedValidShape.push_back(inputValidShape[i]);
    }
    groupedValidShape.push_back(
        inputValidShape[inputValidShape.size() - 2] *
        ((inputValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS));
    return groupedValidShape;
}

std::vector<int64_t> BuildQuantMXScaleShape(const std::vector<int64_t>& inputShape)
{
    auto scaleShape = inputShape;
    scaleShape.back() = CeilDiv(scaleShape.back(), QUANT_MX_SCALE_GROUP_COLS);
    scaleShape.push_back(QUANT_MX_SCALE_PAIR_SIZE);
    return scaleShape;
}

std::vector<SymbolicScalar> BuildQuantMXScaleValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scaleValidShape = inputValidShape;
    scaleValidShape.back() = (scaleValidShape.back() + QUANT_MX_SCALE_GROUP_COLS - 1) / QUANT_MX_SCALE_GROUP_COLS;
    scaleValidShape.push_back(SymbolicScalar(QUANT_MX_SCALE_PAIR_SIZE));
    return scaleValidShape;
}

void CheckQuantMXTileShape(const LogicalTensorPtr& input, const VecTile& vecTile)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX tile shape rank must match input rank.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
        << "QuantMX tile shape last dim must be positive.";

    const int64_t lastDimBytes = vecTile[vecTile.size() - 1] * BytesOf(input->Datatype());
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX tile shape's last dim must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;
}

void TiledQuantMXOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& dst,
    const LogicalTensorPtr& exp, const LogicalTensorPtr& maxScratch, const LogicalTensorPtr& scalingScratch,
    DequantScaleRoundingMode mode, int64_t axis, int64_t performanceMode)
{
    if (cur == input.tensor.GetShape().size()) {
        const int64_t lastDimBytes = input.tileInfo.shape.back() * BytesOf(input.tensor.GetDataType());
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
            << "QuantMX tile width must be 256-byte aligned. Current last dim bytes: " << lastDimBytes;

        auto addQuantMXTile = [&](const std::vector<int64_t>& quantTileShape, const std::vector<int64_t>& tileOffset,
                                  const std::vector<int64_t>& groupedTileShape,
                                  const std::vector<int64_t>& groupedTileOffset) {
            auto srcTile = input.tensor.GetStorage()->View(function, quantTileShape, tileOffset);
            auto dstTile = dst->View(function, quantTileShape, tileOffset);
            auto expTile = exp->View(function, groupedTileShape, groupedTileOffset);
            auto maxTile = maxScratch->View(function, groupedTileShape, groupedTileOffset);
            auto scalingTile = scalingScratch->View(function, quantTileShape, tileOffset);
            auto& tiledOp =
                function.AddOperation(Opcode::OP_QUANT_MX, {srcTile}, {dstTile, expTile, maxTile, scalingTile});
            tiledOp.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(mode));
            tiledOp.SetAttribute(OpAttributeKey::mxQuantAxis, axis);
            tiledOp.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, performanceMode);
        };

        if (performanceMode == 0) {
            auto groupedTileShape = input.tileInfo.shape;
            groupedTileShape.back() = CeilDiv(groupedTileShape.back(), QUANT_MX_GROUP_COLS);
            auto groupedTileOffset = input.tileInfo.offset;
            groupedTileOffset.back() /= QUANT_MX_GROUP_COLS;
            addQuantMXTile(input.tileInfo.shape, input.tileInfo.offset, groupedTileShape, groupedTileOffset);
            return;
        }

        addQuantMXTile(
            input.tileInfo.shape, input.tileInfo.offset, BuildQuantMXPerformanceGroupedShape(input.tileInfo.shape),
            BuildQuantMXPerformanceGroupedOffset(input.tileInfo.offset, input.tensor.GetShape(), input.tileInfo.shape));
        return;
    }

    const auto& vecTile = tileShape.GetVecTile();
    int64_t step = std::max<int64_t>(1, std::min<int64_t>(vecTile[cur], input.tensor.GetShape()[cur]));

    for (int64_t i = 0; i < input.tensor.GetShape()[cur]; i += step) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, step);
        input.tileInfo.offset[cur] = i;
        TiledQuantMXOperation(
            function, tileShape, cur + 1, input, dst, exp, maxScratch, scalingScratch, mode, axis, performanceMode);
    }
}

void QuantMXTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "QuantMX expects 1 input tensor.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 4) << "QuantMX expects 4 output tensors.";

    const auto& src = iOperand[0];
    const auto& dst = oOperand[0];
    const auto& exp = oOperand[1];
    const auto& maxScratch = oOperand[2];
    const auto& scalingScratch = oOperand[3];
    CheckQuantMXDtype(dst->Datatype());
    auto mode = GetQuantMXMode(op);
    auto axis = GetQuantMXAxis(op);
    auto performanceMode = GetQuantMXPerformanceMode(op);
    CheckQuantMXMode(mode);
    CheckQuantMXPerformanceMode(performanceMode);
    CheckQuantMXAxis(axis, src->GetShape().size());
    CheckQuantMXTileShape(src, tileShape.GetVecTile());
    CheckQuantMXPerformanceTileShape(src, tileShape.GetVecTile(), performanceMode);
    TileInfo inputTileInfo(src->shape.size(), src->offset.size());
    auto input = Input{Tensor(src), inputTileInfo};
    TiledQuantMXOperation(
        function, tileShape, 0, input, dst, exp, maxScratch, scalingScratch, mode, axis, performanceMode);
}
} // namespace

// =============================================================================
// Symmetric Quantization (FP32 -> INT8)
// =============================================================================

void TiledQuantizeSymmetric(Function &function, const TileShape &tileShape, size_t cur,
    Input &srcInput, Input &scaleInput, Input &dstInput, int64_t axis) {
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape, scaleInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_QUANTIZE_SYM, {srcTile, scaleTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            // If scale's dimension is 1 (broadcast dimension), use modulo which gives 0
            // Otherwise use the same index as src/dst
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        TiledQuantizeSymmetric(function, tileShape, cur + 1, srcInput, scaleInput, dstInput, axis);
    }
}

void TiledQuantizeSymmetric(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale,
    const LogicalTensorPtr &dst, int64_t axis) {
    ASSERT(src->shape.size() == src->offset.size()) << "Source shape size and offset size should be equal";
    ASSERT(dst->shape.size() == dst->offset.size()) << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeSymmetric(function, tileShape, 0, srcInput, scaleInput, dstInput, axis);
}

// =============================================================================
// Asymmetric Quantization (FP32 -> UINT8)
// =============================================================================

void TiledQuantizeAsymmetric(Function &function, const TileShape &tileShape, size_t cur,
    Input &srcInput, Input &scaleInput, Input &offsetInput, Input &dstInput, int64_t axis) {
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape, scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape, offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {srcTile, scaleTile, offsetTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        // Update offset tile info - offset has same shape as scale
        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledQuantizeAsymmetric(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis);
    }
}

void TiledQuantizeAsymmetric(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, const LogicalTensorPtr &offset,
    const LogicalTensorPtr &dst, int64_t axis) {
    ASSERT(src->shape.size() == src->offset.size()) << "Source shape size and offset size should be equal";
    ASSERT(dst->shape.size() == dst->offset.size()) << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeAsymmetric(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis);
}

// =============================================================================
// Tensor-level Quantization Operations
// =============================================================================

LogicalTensorPtr TensorQuantizeSymmetricOperation(Function &function,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, int64_t axis) {
    // Output is INT8 for symmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT8, src->shape, src->GetDynValidShape());
    auto &op = function.AddOperation(Opcode::OP_QUANTIZE_SYM, {src, scale}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    function.UpdateTensorDataUsage(op);
    return result;
}

LogicalTensorPtr TensorQuantizeAsymmetricOperation(Function &function,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, const LogicalTensorPtr &offset, int64_t axis) {
    // Output is UINT8 for asymmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_UINT8, src->shape, src->GetDynValidShape());
    auto &op = function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    function.UpdateTensorDataUsage(op);
    return result;
}

// =============================================================================
// Public Quantize API
// =============================================================================

Tensor Quantize(const Tensor &input, const Tensor &scale, DataType dtype, int axis, const Tensor &zeroPoints) {
    DECLARE_TRACER();

    // Validate input shapes
    ASSERT(input.GetShape().size() >= SHAPE_DIM2 && input.GetShape().size() <= SHAPE_DIM5)
        << "The shape.size() only support 2~5";

    // Validate data types
    std::vector<DataType> SUPPORT_INPUT_DATATYPES = {DataType::DT_FP32};
    ASSERT(std::find(SUPPORT_INPUT_DATATYPES.begin(), SUPPORT_INPUT_DATATYPES.end(), input.GetDataType()) !=
           SUPPORT_INPUT_DATATYPES.end())
        << "The input datatype is not supported";

    // Normalize axis to negative indexing
    int ndim = static_cast<int>(input.GetShape().size());
    int normalizedAxis = axis;
    if (axis >= 0) {
        normalizedAxis = axis - ndim;
    }
    ASSERT(normalizedAxis == -1 || normalizedAxis == -2) << "Only axis=-1 (per-row) and axis=-2 (per-column) are supported";

    // Determine quantization type based on zeroPoints presence
    bool isAsymmetric = (zeroPoints.GetStorage() != nullptr);

    // For axis=-2 (per-column quantization), use Transpose to swap last two dimensions
    // Strategy: transpose input and scale -> quantize with axis=-1 -> transpose output back
    if (normalizedAxis == -2) {
        // Swap last two dimensions: [..., H, W] -> [..., W, H]
        int lastDim = ndim - 1;        // -1 in positive index
        int secondLastDim = ndim - 2;  // -2 in positive index

        // Transpose input: [..., H, W] -> [..., W, H]
        Tensor transposedInput = Transpose(input, {secondLastDim, lastDim});
        // [TQuant] get tmp Tile for Tquant
        VecTile oriVectile = TileShape::Current().GetVecTile();
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        std::swap(tmpVectile[secondLastDim], tmpVectile[lastDim]);
        TileShape::Current().SetVecTile(tmpVectile);
        // [TQuant] get tmp validShape
        auto tmpValidShape = input.GetStorage()->dynValidShape_;
        std::swap(tmpValidShape[secondLastDim], tmpValidShape[lastDim]);
        transposedInput.GetStorage()->UpdateDynValidShape(tmpValidShape);
        
        // init result
        Tensor quantizedResult;

        if (isAsymmetric) {
            // Asymmetric quantization with axis=-1
            ASSERT(dtype == DataType::DT_UINT8)
                << "Asymmetric quantization output type should be UINT8";
            quantizedResult = CALL(QuantizeAsymmetricOperation,
                *Program::GetInstance().GetCurrentFunction(),
                transposedInput.GetStorage(), scale.GetStorage(),
                zeroPoints.GetStorage(), normalizedAxis);
        } else {
            // Symmetric quantization with axis=-1
            ASSERT(dtype == DataType::DT_INT8)
                << "Symmetric quantization output type should be INT8";
            quantizedResult = CALL(QuantizeSymmetricOperation,
                *Program::GetInstance().GetCurrentFunction(),
                transposedInput.GetStorage(), scale.GetStorage(), normalizedAxis);
        }

        // [Transpose] set tmp validShape
        quantizedResult.GetStorage()->UpdateDynValidShape(tmpValidShape);
        // [Transpose] set tmp VecTile
        TileShape::Current().SetVecTile(tmpVectile);
        // output back: [..., W, H] -> [..., H, W]
        Tensor result = Transpose(quantizedResult, {secondLastDim, lastDim});
        // [Tstore] get origin ValidShape
        result.GetStorage()->UpdateDynValidShape(input.GetStorage()->dynValidShape_);
        // [Tstore] get origin TileShape
        TileShape::Current().SetVecTile(oriVectile);
        return result;
    }

    // axis=-1 case: direct quantization without transpose
    if (isAsymmetric) {
        // Asymmetric quantization: FP32 -> UINT8
        ASSERT(dtype == DataType::DT_UINT8)
            << "Asymmetric quantization output type should be UINT8";
        RETURN_CALL(QuantizeAsymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
            input.GetStorage(), scale.GetStorage(), zeroPoints.GetStorage(), normalizedAxis);
    } else {
        // Symmetric quantization: FP32 -> INT8
        ASSERT(dtype == DataType::DT_INT8)
            << "Symmetric quantization output type should be INT8";
        RETURN_CALL(QuantizeSymmetricOperation, *Program::GetInstance().GetCurrentFunction(),
            input.GetStorage(), scale.GetStorage(), normalizedAxis);
    }
}

// =============================================================================
// Tile Function Registration
// =============================================================================

void QuantizeSymmetricOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledQuantizeSymmetric(function, tileShape, iOperand[0], iOperand[1], oOperand[0], axis);
}

void QuantizeAsymmetricOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledQuantizeAsymmetric(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis);
}

REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, QuantizeSymmetricOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, QuantizeAsymmetricOperationTileFunc);

// =============================================================================
// Dequantization Operations (INT8/INT16 -> FP32)
// TDequant always requires 4 params: dst, src, scale, offset (symmetric: offset=0)
// =============================================================================

void TiledDequantize(Function &function, const TileShape &tileShape, size_t cur,
    Input &srcInput, Input &scaleInput, Input &offsetInput, Input &dstInput, int64_t axis) {
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape, scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape, offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto &op = function.AddOperation(Opcode::OP_DEQUANTIZE, {srcTile, scaleTile, offsetTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledDequantize(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis);
    }
}

void TiledDequantize(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, const LogicalTensorPtr &offset,
    const LogicalTensorPtr &dst, int64_t axis) {
    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledDequantize(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis);
}

LogicalTensorPtr TensorDequantizeOperation(Function &function,
    const LogicalTensorPtr &src, const LogicalTensorPtr &scale, const LogicalTensorPtr &offset, int64_t axis) {
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, src->shape, src->GetDynValidShape());
    auto &op = function.AddOperation(Opcode::OP_DEQUANTIZE, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    function.UpdateTensorDataUsage(op);
    return result;
}

// Helper: create zero tensor for symmetric dequantization
static LogicalTensorPtr CreateZeroOffsetTensor(Function &function, const LogicalTensorPtr &scale) {
    Element zeroVal(DataType::DT_FP32, (int64_t)0);

    Tensor zeroTensor = TensorFullOperation(function, zeroVal, SymbolicScalar(),
        DataType::DT_FP32 ,scale->shape, scale->GetDynValidShape());

    return zeroTensor.GetStorage();
}

// Public Dequantize API
Tensor Dequantize(const Tensor &input, const Tensor &scale, DataType otype, int axis, const Tensor &zeroPoints) {
    DECLARE_TRACER();

    // Validate input shapes: 2D to 5D tensors supported
    size_t inputRank = input.GetShape().size();
    ASSERT(inputRank >= SHAPE_DIM2 && inputRank <= SHAPE_DIM5)
        << "Dequantize input rank must be 2~5, but got rank=" << inputRank;

    // Validate input data type: INT8 or INT16
    ASSERT(input.GetDataType() == DataType::DT_INT8 || input.GetDataType() == DataType::DT_INT16)
        << "Dequantize input dtype must be INT8 or INT16, but got dtype="
        << static_cast<int>(input.GetDataType());

    // Validate output type
    ASSERT(otype == DataType::DT_FP32)
        << "Dequantize output type must be FP32, but got dtype=" << static_cast<int>(otype);

    // Normalize axis to negative indexing and validate
    int ndim = static_cast<int>(input.GetShape().size());
    int normalizedAxis = axis;
    if (axis >= 0) {
        normalizedAxis = axis - ndim;
    }
    ASSERT(normalizedAxis == -1 || normalizedAxis == -2)
        << "Dequantize axis must be -1 (per-row) or -2 (per-column), but got axis="
        << axis << " (normalized=" << normalizedAxis << ")";

    // Determine if symmetric or asymmetric
    bool isAsymmetric = (zeroPoints.GetStorage() != nullptr);

    // For axis=-2, use Transpose (consistent with Quantize implementation)
    if (normalizedAxis == -2) {
        int lastDim = ndim - 1;
        int secondLastDim = ndim - 2;

        // Transpose input: [..., H, W] -> [..., W, H]
        Tensor transposedInput = Transpose(input, {secondLastDim, lastDim});
        // [TDequant] get tmp Tile for TDequant
        VecTile oriVectile = TileShape::Current().GetVecTile();
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        std::swap(tmpVectile[secondLastDim], tmpVectile[lastDim]);
        TileShape::Current().SetVecTile(tmpVectile);
        // [TDequant] get tmp validShape
        auto tmpValidShape = input.GetStorage()->dynValidShape_;
        std::swap(tmpValidShape[secondLastDim], tmpValidShape[lastDim]);
        transposedInput.GetStorage()->UpdateDynValidShape(tmpValidShape);

        // init result
        Tensor dequantizedResult;

        if (isAsymmetric) {
            // scale and zeroPoints are NOT transposed (consistent with Quantize)
            dequantizedResult = TensorDequantizeOperation(
                *Program::GetInstance().GetCurrentFunction(),
                transposedInput.GetStorage(), scale.GetStorage(),
                zeroPoints.GetStorage(), -1);
        } else {
            // Symmetric: create zero offset tensor
            auto zeroOffset = CreateZeroOffsetTensor(*Program::GetInstance().GetCurrentFunction(),
                scale.GetStorage());
            dequantizedResult = TensorDequantizeOperation(
                *Program::GetInstance().GetCurrentFunction(),
                transposedInput.GetStorage(), scale.GetStorage(),
                zeroOffset, -1);
        }

        // [Transpose] set tmp validShape
        dequantizedResult.GetStorage()->UpdateDynValidShape(tmpValidShape);
        // [Transpose] set tmp VecTile
        TileShape::Current().SetVecTile(tmpVectile);
        // output back: [..., W, H] -> [..., H, W]
        Tensor result = Transpose(dequantizedResult, {secondLastDim, lastDim});
        // [Tstore] get origin ValidShape
        result.GetStorage()->UpdateDynValidShape(input.GetStorage()->dynValidShape_);
        // [Tstore] get origin TileShape
        TileShape::Current().SetVecTile(oriVectile);
        return result;
    }

    // axis=-1 case
    if (isAsymmetric) {
        RETURN_CALL(DequantizeOperation, *Program::GetInstance().GetCurrentFunction(),
            input.GetStorage(), scale.GetStorage(), zeroPoints.GetStorage(), normalizedAxis);
    } else {
        // Symmetric: create zero offset tensor
        auto zeroOffset = CreateZeroOffsetTensor(*Program::GetInstance().GetCurrentFunction(),
            scale.GetStorage());
        RETURN_CALL(DequantizeOperation, *Program::GetInstance().GetCurrentFunction(),
            input.GetStorage(), scale.GetStorage(), zeroOffset, normalizedAxis);
    }
}

Tensor DequantizeSymmetric(const Tensor &src, const Tensor &scale, int64_t axis) {
    return Dequantize(src, scale, DataType::DT_FP32, axis, Tensor());
}

Tensor DequantizeAsymmetric(const Tensor &src, const Tensor &scale, const Tensor &zeroPoints, int64_t axis) {
    return Dequantize(src, scale, DataType::DT_FP32, axis, zeroPoints);
}

std::tuple<Tensor, Tensor> QuantMX(
    const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis, bool performanceMode)
{
    DECLARE_TRACER();
    CheckQuantMXPerformanceMode(static_cast<int64_t>(performanceMode));
    CheckQuantMXInput(input, quantDtype, mode, axis);
    const auto oldVecTile = TileShape::Current().GetVecTile();
    if (performanceMode && !oldVecTile.tile.empty()) {
        CheckQuantMXPerformanceTileShape(input.GetStorage(), oldVecTile, static_cast<int64_t>(performanceMode));
    }

    const auto& inputShape = input.GetShape();
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, inputShape.size());
    const std::vector<int64_t> groupedShape = performanceMode ? BuildQuantMXPerformanceGroupedShape(inputShape) :
                                                                BuildQuantMXGroupedShape(inputShape);
    const std::vector<int64_t> scaleShape = BuildQuantMXScaleShape(inputShape);

    const auto scratchDtype = input.GetDataType();
    auto quantized = Tensor(quantDtype, inputShape, "", TileOpFormat::TILEOP_ND);
    auto exp = Tensor(DataType::DT_FP8E8M0, groupedShape, "", TileOpFormat::TILEOP_ND);
    auto maxScratch = Tensor(scratchDtype, groupedShape, "", TileOpFormat::TILEOP_ND);
    auto scalingScratch = Tensor(scratchDtype, inputShape, "", TileOpFormat::TILEOP_ND);

    std::vector<SymbolicScalar> scaleValidShape;
    const auto& inputValidShape = input.GetStorage()->GetDynValidShape();
    if (!inputValidShape.empty()) {
        quantized.GetStorage()->UpdateDynValidShape(inputValidShape);
        const auto groupedValidShape = performanceMode ? BuildQuantMXPerformanceGroupedValidShape(inputValidShape) :
                                                         BuildQuantMXGroupedValidShape(inputValidShape);
        exp.GetStorage()->UpdateDynValidShape(groupedValidShape);
        maxScratch.GetStorage()->UpdateDynValidShape(groupedValidShape);
        scalingScratch.GetStorage()->UpdateDynValidShape(inputValidShape);
        scaleValidShape = BuildQuantMXScaleValidShape(inputValidShape);
    }

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_QUANT_MX, {input.GetStorage()},
        {quantized.GetStorage(), exp.GetStorage(), maxScratch.GetStorage(), scalingScratch.GetStorage()});
    op.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(mode));
    op.SetAttribute(OpAttributeKey::mxQuantAxis, normalizedAxis);
    op.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, static_cast<int64_t>(performanceMode ? 1 : 0));
    if (performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(BuildQuantMXPerformanceVecTile(oldVecTile.tile));
    }
    auto scale = Reshape(exp, scaleShape, scaleValidShape);
    if (performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(oldVecTile);
    }
    return std::tie(quantized, scale);
}

// Tile Function Registration
void DequantizeOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledDequantize(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis);
}

REGISTER_OPERATION_TILED_FUNC(OP_DEQUANTIZE, Opcode::OP_DEQUANTIZE, DequantizeOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(QuantMX, Opcode::OP_QUANT_MX, QuantMXTileFunc);

} // namespace npu::tile_fwk
