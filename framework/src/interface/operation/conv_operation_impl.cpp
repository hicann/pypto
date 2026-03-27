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
 * \file conv_operation_impl.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"
#include "interface/utils/conv_error.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
namespace Conv {

const std::string L12L0ConvOpAttributeKey::postK = "POST_K";
const std::string L12L0ConvOpAttributeKey::postM = "POST_M";
const std::string L12L0ConvOpAttributeKey::postN = "POST_N";
const std::string L12L0ConvOpAttributeKey::filterH = "FILTER_H";
const std::string L12L0ConvOpAttributeKey::filterW = "FILTER_W";
const std::string L12L0ConvOpAttributeKey::strideH = "STRIDE_H";
const std::string L12L0ConvOpAttributeKey::strideW = "STRIDE_W";
const std::string L12L0ConvOpAttributeKey::dilationH = "DILATION_H";
const std::string L12L0ConvOpAttributeKey::dilationW = "DILATION_W";
const std::string L12L0ConvOpAttributeKey::paddingLeft = "PAD_LEFT";
const std::string L12L0ConvOpAttributeKey::paddingRight = "PAD_RIGHT";
const std::string L12L0ConvOpAttributeKey::paddingTop = "PAD_TOP";
const std::string L12L0ConvOpAttributeKey::paddingBottom = "PAD_BOTTOM";
const std::string L12L0ConvOpAttributeKey::padValue = "PAD_VALUE";
const std::string LoadStoreConvOpAttributeKey::copyInMode = "COPY_IN_MODE";
const std::string LoadStoreConvOpAttributeKey::copyOutMode = "COPY_OUT_MODE";
const std::string LoadStoreConvOpAttributeKey::isFmap = "IS_FMAP";
const std::string LoadStoreConvOpAttributeKey::isConv3D = "IS_CONV3D";

std::vector<int64_t> rotateVector(const std::vector<int64_t>& input, size_t shift) {
    std::vector<int64_t> result = input;
    std::rotate(result.begin(), result.begin() + shift, result.end());
    return result;
}

void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max, const std::string& formula = "")
{
    std::ostringstream oss;
    oss << "Invalid " << name << ":" << value
        << ", expected range [" << min << "," << max << "].";
    if (!formula.empty()) {
        oss << "Formula: " << formula;
    }
    ASSERT(ConvOperationError::INPUT_INVALID, value >= min && value <= max) << oss.str();
}

int64_t ConvComputeHo(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    if (attrParam.isConv1D) {
        return 1;
    }
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideH = strides[PAD_STRIDE_H];
    if (strideH == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padTop = paddings[PAD_TOP_INDEX];
    int64_t padBottom = paddings[PAD_BOTTOM_INDEX];
    int64_t dilationH = dilations[PAD_STRIDE_H];
    int64_t hin = inputTensor.GetShape()[indexH];
    int64_t kh = weightTensor.GetShape()[indexH];
    int64_t cmpHo = (hin + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
    return cmpHo;
}

int64_t ConvComputeWo(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    uint32_t indexAttr = attrParam.isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W;

    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideW = strides[indexAttr];
    if (strideW == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t dilationW = dilations[indexAttr];
    int64_t padLeft = paddings[2 * indexAttr];
    int64_t padRight = paddings[2* indexAttr + 1];
    int64_t win = inputTensor.GetShape()[indexW];
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t cmpWo = (win + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
    return cmpWo;
}

int64_t ConvComputeDo(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    std::vector<int64_t> strides = attrParam.strides;
    int64_t strideD = strides[PAD_STRIDE_D];
    if (strideD == 0) {
        return 1;
    }
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    int64_t padHead = paddings[PAD_HEAD_INDEX];
    int64_t padTail = paddings[PAD_TAIL_INDEX];
    int64_t dilationD = dilations[PAD_STRIDE_D];
    int64_t din = inputTensor.GetShape()[NCDHW_D_IDX];
    int64_t kd = weightTensor.GetShape()[NCDHW_D_IDX];
    int64_t cmpDo = (din + padHead + padTail - dilationD * (kd - 1) - 1) / strideD + 1;
    return cmpDo;
}

void CheckOutputShape(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    std::string hOutFormula = "hOut = (hin + 2 * pad_h - (kh - 1) * dilation_h - 1) / stride_h + 1";
    CheckValueRange(hOut, "hOut" , NUM1, MAX_SIZE, hOutFormula);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    std::string wOutFormula = "wOut = (win + 2 * pad_w - (kw - 1) * dilation_w - 1) / stride_w + 1";
    CheckValueRange(wOut, "wOut" , NUM1, MAX_SIZE, wOutFormula);
    if (attrParam.isConv3D) {
        int64_t dOut = ConvComputeDo(inputTensor, weightTensor, attrParam);
        std::string dOutFormula = "dOut = (din + 2 * pad_d - (kd - 1) * dilation_d - 1) / stride_d + 1";
        CheckValueRange(dOut, "dOut" , NUM1, MAX_SIZE, dOutFormula);
    }
}

void CheckAlignment(int64_t value, int64_t alignment, const std::string& valueName)
{
    ASSERT(ConvOperationError::INPUT_INVALID, alignment != 0)
        << "Error in alignment check for "<< valueName << ".";
    ASSERT(ConvOperationError::INPUT_INVALID, value % alignment == 0)
        << "Invalid " << valueName << ":" << value << ", requires " << alignment << "-element alignment.";
}

int64_t ConvAlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

void CheckHowoTile(const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t tileHout = convTile.tileL1Info.tileHout;
    int64_t tileWout = convTile.tileL1Info.tileWout;
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, attrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, attrParam);
    if (wOut % 16 != 0) {
        ASSERT(ConvOperationError::INPUT_INVALID, tileHout == 1)
            << "When wOut is not a multiple of 16, tileHout should be 1.";
    }
    CheckValueRange(tileHout, "tileHout" , NUM1, hOut);
    if (tileHout > 1) {
        ASSERT(ConvOperationError::INPUT_INVALID, tileWout == wOut)
            << "When tileHout > 1, tileWout must be equal to wOut.Now tileHout=" << tileHout
            << ", tileWout=" << tileWout << ", wOut=" << wOut;
    }
    CheckValueRange(tileWout, "tileWout" , NUM1, ConvAlignB(wOut, NUM16));
    CheckAlignment(tileWout, NUM16, "tileWout");
}

void ValidateL0Constraint(int64_t tile1, int64_t tile2, int64_t tile3, size_t dtypeSize, size_t cacheSize, const std::string& cacheName,
    const std::string& dim1Name, const std::string& dim2Name, const std::string& dim3Name)
{
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, tile1 * tile2 * tile3 * dtypeSize <= cacheSize)
        << "Shape does not satisfy " << cacheName
        << " load constraints, " << dim1Name << ":" << tile1
        << ", " << dim2Name << ":" << tile2 << ", " << dim3Name << ":" << tile3
        << ", which must satisfy " << dim1Name << " × " << dim2Name << " × "
        << dim3Name << " × dtypesize ≤ " << cacheName << "Size(" << cacheSize << ").";
}

void CheckL0TileTiling(DataType outType, const ConvAttrParam &attrParam, const Tensor &weightTensor, const Tensor &inputTensor)
{
    auto &convTile = TileShape::Current().GetConvTile();
    int64_t tileH = convTile.tileL0Info.tileH, tileW = convTile.tileL0Info.tileW;
    int64_t tileN = convTile.tileL0Info.tileN, tileK = convTile.tileL0Info.tileK;
    int64_t tileHout = convTile.tileL1Info.tileHout, tileWout = convTile.tileL1Info.tileWout;
    int64_t tileCout = convTile.tileL1Info.tileN, k0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t cin = inputTensor.GetShape()[NCHW_C_IDX];
    int64_t cout = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t kAL1 = ConvAlignB(tileCinFmap, k0) * kh * kw, oriK = ConvAlignB(cin, k0) * kh * kw;
    int64_t kBL1 = ConvAlignB(tileCinWeight, k0) * kh * kw;
    int64_t batch = inputTensor.GetShape()[NCHW_N_IDX], groups = attrParam.groups;
    int numTileL0 = batch * groups * CeilDiv(cout/groups, tileN) * CeilDiv(tileHout, tileH) * CeilDiv(tileWout, tileW);
    if (attrParam.isConv3D) {
        int64_t kd = weightTensor.GetShape()[NCDHW_D_IDX];
        int64_t dout = ConvComputeDo(inputTensor, weightTensor, attrParam);
        numTileL0 *= dout; kAL1 *= kd; kBL1 *= kd; oriK *= kd;
    }
    if (numTileL0 > MAX_LOOP || CeilDiv(oriK, tileK) > MAX_LOOP) {
        CONV_LOGW("Suggestion: Consider increasing tile size to reduce compilation time.");
    }
    int64_t minKL1 = std::min(kAL1, kBL1);
    CheckAlignment(tileK , k0, "tileK");
    CheckValueRange(tileH, "tileH" , NUM1, tileHout);
    CheckValueRange(tileW, "tileW" , NUM1, tileWout);
    CheckValueRange(tileK, "tileK" , NUM1, minKL1);
    CheckAlignment(tileN, NUM16, "tileL0Info.tileN");
    CheckAlignment(tileW, NUM16, "tileW");
    CheckValueRange(tileN, "tileL0Info.tileN" , NUM1, ConvAlignB(tileCout, NUM16));
    ASSERT(ConvOperationError::INPUT_INVALID, kAL1 % tileK == 0 && kBL1 % tileK == 0)
        << "Invalid tileK: " << tileK << ", must be a factor of both kAL1:" << kAL1 << " and kBL1:" << kBL1;
    Platform& platform = Platform::Instance();
    size_t l0aSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0A);
    size_t l0bSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0B);
    size_t l0cSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0C);
    ValidateL0Constraint(tileH, tileW, tileK, BytesOf(outType), l0aSize, "L0A", "tileH", "tileW", "tileK");
    ValidateL0Constraint(tileK, tileN, 1, BytesOf(outType), l0bSize, "L0B", "tileK", "tileN", "");
    ValidateL0Constraint(tileH, tileW, tileN, BytesOf(DataType::DT_FP32), l0cSize, "L0C", "tileH", "tileW", "tileN");
}

void CheckDivisible(int64_t value, int64_t divisor, const std::string& valueName, const std::string& divisorName)
{
    ASSERT(ConvOperationError::INPUT_INVALID, divisor != 0) << divisorName << " cannot be zero.";
    ASSERT(ConvOperationError::INPUT_INVALID, value % divisor == 0)
        << "The value of " << divisorName << " (" << divisor
        << ") does not divide "<< valueName
        << "(" << value << "). Adjusting " << divisorName
        << " to the nearest value such that "<< valueName
        << " % " << divisorName << " == 0.";
}

void CheckTileTiling(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    auto convTile = TileShape::Current().GetConvTile();
    int64_t tileHin = convTile.tileL1Info.tileHin;
    int64_t tileWin = convTile.tileL1Info.tileWin;
    int64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    int64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    int64_t tileN = convTile.tileL1Info.tileN;
    int64_t tileBatch = convTile.tileL1Info.tileBatch;
    int64_t groups = attrParam.groups;

    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t hin = attrParam.isConv1D ? 1 : inputTensor.GetShape()[indexH];
    int64_t win = inputTensor.GetShape()[indexW];

    CheckValueRange(tileHin, "tileHin", NUM1, hin);
    CheckValueRange(tileBatch, "tileBatch", NUM1, NUM1);
    CheckValueRange(tileWin, "tileWin", NUM1, win);
    CheckValueRange(tileN, "tileL1Info.tileN", NUM1, ConvAlignB(cOut/groups, NUM16));
    CheckAlignment(tileN, NUM16, "tileL1Info.tileN");

    CheckHowoTile(inputTensor, weightTensor, attrParam);
    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    CheckAlignment(tileCinFmap, k0, "tileCinFmap");
    CheckAlignment(tileCinWeight, k0, "tileCinWeight");
    if (convTile.setL0Tile){
        CheckL0TileTiling(outType, attrParam, weightTensor, inputTensor);
    }
}

uint64_t Conv2DInferHiL1(uint64_t inputHoL1, uint64_t khDilated, uint64_t hi, uint64_t strideH)
{
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }
    return tmpHiL1;
}

void CheckL1SizeTiling(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor, ConvAttrParam &attrParam)
{
    auto convTile = TileShape::Current().GetConvTile();
    Platform& platform = Platform::Instance();
    uint64_t l1Size = platform.GetAIVCore().GetMemorySize(MemoryType::MEM_L1);
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);

    uint64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    uint64_t hin = attrParam.isConv1D ? 1 : inputTensor.GetShape()[indexH];
    uint64_t kw = weightTensor.GetShape()[indexW];
    uint64_t win = inputTensor.GetShape()[indexW];
    uint64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);

    std::vector<int64_t> strides = attrParam.strides;
    std::vector<int64_t> dilations = attrParam.dilations;
    uint32_t indexAttrW = attrParam.isConv1D ? PAD_STRIDE_H : PAD_STRIDE_W;
    uint64_t strideH = attrParam.isConv1D ? 1 : strides[PAD_STRIDE_H];
    uint64_t strideW = strides[indexAttrW];
    uint64_t dilationH = attrParam.isConv1D ? 1 : dilations[PAD_STRIDE_H];
    uint64_t dilationW = dilations[indexAttrW];

    uint64_t biasL1Size = 0;
    uint64_t tileN = convTile.tileL1Info.tileN;
    if (!biasTensor.IsEmpty()) {
        biasL1Size = ConvAlignB(tileN * BytesOf(outType), ALIGN_SIZE_32);
    }
    uint64_t tileCinFmap = convTile.tileL1Info.tileCinFmap;
    uint64_t tileCinWeight = convTile.tileL1Info.tileCinWeight;
    uint64_t kBL1 = ConvAlignB(tileCinWeight * kh * kw, k0);
    uint64_t weightL1Size = ConvAlignB(kBL1 * tileN * BytesOf(outType), ALIGN_SIZE_32);

    uint64_t inputL1Size = 0;
    uint64_t tileWout = convTile.tileL1Info.tileWout;
    uint64_t tileHout = convTile.tileL1Info.tileHout;
    uint64_t khDilated = (kh - 1) * dilationH + 1;
    uint64_t hiAL1 = std::min((tileHout - 1) * strideH + khDilated, hin);
    uint64_t kwDilated = (kw - 1) * dilationW + 1;
    uint64_t wiAL1 = std::min((tileWout - 1) * strideW + kwDilated, win);;

    inputL1Size = ConvAlignB(hiAL1 * wiAL1 * tileCinFmap * BytesOf(outType), ALIGN_SIZE_32);
    uint64_t minL1LoadSize = biasL1Size + inputL1Size + weightL1Size;
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, minL1LoadSize <= l1Size)
        << "MinL1LoadSize > L1size, current MinL1LoadSize: " << minL1LoadSize << ", L1size: " << l1Size << ".";
}

void CheckGroupsShape(const int64_t cinFmap, const int64_t cinWeight,const int64_t cOut, const int64_t groups)
{
    CheckValueRange(groups, "groups", NUM1, SHAPE_INNER_AXIS_MAX_SIZE);

    CheckDivisible(cinFmap, groups, "Cin", "groups");
    CheckDivisible(cOut, groups, "Cout", "groups");

    ASSERT(ConvOperationError::INPUT_INVALID, cinFmap == cinWeight * groups)
        << "Fmap Cin (" << cinFmap << ") != weight Cin (" << cinWeight << ") * groups (" << groups << ").";
}

void CheckDimParam(const std::vector<int64_t>& vec, const std::string& name, int expectedDim)
{
    ASSERT(ConvOperationError::INPUT_INVALID, vec.size() == static_cast<size_t>(expectedDim))
        << "Input attr " << name << " dim: " << vec.size() << " != " << expectedDim << ".";
}

void CheckDimensionRange(const std::vector<int64_t>& vec, const std::string& name, int minVal, int maxVal)
{
    for (size_t i = 0; i < vec.size(); ++i) {
        ASSERT(ConvOperationError::INPUT_INVALID, vec[i] >= minVal && vec[i] <= maxVal)
            << "The value of the " << i << "-th dimension of " << name << " must be in the range [" << minVal
            << "," << maxVal << "].Current value:" << vec[i] << ".";
    }
}

void CheckLoad3dShape(DataType outType, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    std::vector<int64_t> paddings = attrParam.paddings;
    std::vector<int64_t> dilations = attrParam.dilations;
    std::vector<int64_t> strides = attrParam.strides;
    if (attrParam.isConv3D) {
        paddings = rotateVector(paddings, 4);
        dilations = rotateVector(dilations, 2);
        strides = rotateVector(strides, 2);
    }
    CheckDimensionRange(paddings, "paddings", 0, MAX_PAD_KERNEL);
    CheckDimensionRange(dilations, "dilations", NUM1, MAX_DILATION_STRIDE);
    CheckDimensionRange(strides, "strides", NUM1, MAX_DILATION_STRIDE);

    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCHW_H_IDX : NCHW_W_IDX);
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    ASSERT(ConvOperationError::INPUT_INVALID, kh <= MAX_PAD_KERNEL && kw  <= MAX_PAD_KERNEL)
        << "Weight shapes do not satisfy Load3D's"
        << (attrParam.isConv1D ? " limit: kw=" : " limits: kh=")
        << (attrParam.isConv1D ? kw : kh)
        << (attrParam.isConv1D ? "" : ", kw=" + std::to_string(kw))
        << ", which must <= " << MAX_PAD_KERNEL << ".";

    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType);
    ASSERT(ConvOperationError::INPUT_INVALID, kh * kw * k0 <= SHAPE_INNER_AXIS_MAX_SIZE)
        << "Weight shapes do not satisfy Load3D's limits: kh*kw*k0=" << kh * kw * k0
        << "(k0 = 32 bytes / dtypesize), which must <=" << SHAPE_INNER_AXIS_MAX_SIZE << ".";
}

void CheckAttrShape(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const ConvAttrParam &attrParam)
{
    std::vector<int64_t> paddings = attrParam.paddings;
    uint32_t index = attrParam.isConv3D ? SHAPE_DIM3 : (attrParam.isConv1D ? SHAPE_DIM1 : SHAPE_DIM2);
    CheckDimParam(attrParam.paddings, "paddings", index * 2);
    CheckDimParam(attrParam.dilations, "dilations", index);
    CheckDimParam(attrParam.strides, "strides", index);
    int64_t groups = attrParam.groups;
    int64_t cinFmap = inputTensor.GetShape()[NCHW_C_IDX];
    int64_t cinWeight = weightTensor.GetShape()[NCHW_C_IDX];
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];

    if (attrParam.isConv3D) {
        paddings = rotateVector(paddings, 4);
    }
    const std::vector<std::string> dimNames =
        attrParam.isConv1D ? std::vector<std::string>{"L"} :
        attrParam.isConv3D ? std::vector<std::string>{"D", "H", "W"} :
        std::vector<std::string>{"H", "W"};
    for (size_t i = 0; i < paddings.size() / 2; ++i) {
        int weightVal = weightTensor.GetShape()[i + 2];
        int paddingLeft = paddings[i * 2];
        int paddingRight = paddings[i * 2 + 1];
        ASSERT(ConvOperationError::INPUT_INVALID, paddingLeft < weightVal && paddingRight < weightVal)
            << "The value of the " << dimNames[i]
            << " dimension of weight must be > padding.Current weight value:" << weightVal
            << ",padding value:" << paddingLeft
            << " and " << paddingRight
            << ".";
    }
    CheckGroupsShape(cinFmap, cinWeight, cOut, groups);
    CheckLoad3dShape(outType, weightTensor, attrParam);
}

void CheckOriginShape(const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor)
{
    CheckDimensionRange(inputTensor.GetShape(), "fmap", NUM1, MAX_SIZE);
    CheckDimensionRange(weightTensor.GetShape(), "weight", NUM1, MAX_SIZE);

    if (biasTensor.IsEmpty()) {
        return;
    }
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    ASSERT(ConvOperationError::INPUT_INVALID, biasTensor.GetShape()[0] == cOut) <<
        "Input illegal bias shape:" << biasTensor.GetShape()[0] << ", which must equal to Cout:" << cOut << ".";
}

void CheckConvOperands(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const Tensor &biasTensor, ConvAttrParam &attrParam)
{
    ASSERT(ConvOperationError::INPUT_INVALID,
        outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    if (inputTensor.Dim() == CONV1D_INPUT_DIM && weightTensor.Dim() == CONV1D_INPUT_DIM) {
        attrParam.isConv1D = true;
    } else if (inputTensor.Dim() == CONV3D_INPUT_DIM && weightTensor.Dim() == CONV3D_INPUT_DIM) {
        attrParam.isConv3D = true;
    }
    CheckOriginShape(inputTensor, weightTensor, biasTensor);
    CheckOutputShape(inputTensor, weightTensor, attrParam);
    CheckAttrShape(outType, inputTensor, weightTensor, attrParam);
    CheckTileTiling(outType, inputTensor, weightTensor, attrParam);
    CheckL1SizeTiling(outType, inputTensor, weightTensor, biasTensor, attrParam);
}

void SetTensorOpAttr(Operation &op, const LogicalTensorPtr &inputTensor, const LogicalTensorPtr &weightTensor,
                     const LogicalTensorPtr &resTensor, const ConvAttrParam &convAttrParam)
{
    op.SetAttribute(CONV_BIAS_ATTR, convAttrParam.hasBias);
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
    op.SetAttribute(CONV_3D_FLAG, convAttrParam.isConv3D);
    op.SetAttribute(CONV_ORI_FMAP_SHAPE_ATTR, inputTensor->GetShape());
    op.SetAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR, weightTensor->GetShape());
    op.SetAttribute(CONV_ORI_RES_SHAPE_ATTR, resTensor->GetShape());
}

Tensor ConstructTensorGraph(const Tensor &inputTensor, const Tensor &weightTensor,
                            const Tensor &biasTensor, const Tensor &resTensor, ConvAttrParam &convAttrParam)
{
    // add Conv node
    Function *functionPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, functionPtr != nullptr)
        << "functionPtr is nullptr.";
    std::vector<LogicalTensorPtr> operandVecIn = {inputTensor.GetStorage(), weightTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    if (convAttrParam.isConv1D) {
        // conv1d case, unsqueeze input to NC1W
        std::vector<int64_t> fmap4DimShape{inputTensor.GetShape()[NCHW_N_IDX], inputTensor.GetShape()[NCHW_C_IDX],
                                           1, inputTensor.GetShape()[NCHW_H_IDX]};
        Tensor fmap4DimTensor(inputTensor.GetStorage()->Datatype(), fmap4DimShape, "", inputTensor.Format());
        std::vector<int64_t> weight4DimShape{weightTensor.GetShape()[NCHW_N_IDX], weightTensor.GetShape()[NCHW_C_IDX],
                                             1, weightTensor.GetShape()[NCHW_H_IDX]};
        Tensor weigth4DimTensor(weightTensor.GetStorage()->Datatype(), weight4DimShape, "", weightTensor.Format());
        auto &reshapeFmapOp =
            functionPtr->AddOperation(Opcode::OP_RESHAPE, {inputTensor.GetStorage()}, {fmap4DimTensor.GetStorage()});
        auto &reshapeWeightOp =
            functionPtr->AddOperation(Opcode::OP_RESHAPE,
                                      {weightTensor.GetStorage()}, {weigth4DimTensor.GetStorage()});
        reshapeFmapOp.SetAttribute("isConv", true);
        reshapeWeightOp.SetAttribute("isConv", true);
        operandVecIn = {fmap4DimTensor.GetStorage(), weigth4DimTensor.GetStorage()};
    }
    if (!biasTensor.IsEmpty()) {
        convAttrParam.hasBias = true;
        std::vector<int64_t> bias2DimShape{1, biasTensor.GetShape()[0]};
        Tensor bias2DimTensor(biasTensor.GetStorage()->Datatype(), bias2DimShape, "", biasTensor.Format());
        auto &reshapeBiasOp =
            functionPtr->AddOperation(Opcode::OP_RESHAPE, {biasTensor.GetStorage()}, {bias2DimTensor.GetStorage()});
        reshapeBiasOp.SetAttribute("isConv", true);
        operandVecIn.push_back(bias2DimTensor.GetStorage());
    }

    if (convAttrParam.isConv1D) {
        // conv1d case, squeeze output to NCL
        std::vector<int64_t> res4DimShape{inputTensor.GetShape()[NCHW_N_IDX], weightTensor.GetShape()[NCHW_N_IDX],
                                          1, resTensor.GetShape()[NCHW_H_IDX]};
        Tensor res4DimTensor(resTensor.GetStorage()->Datatype(), res4DimShape, "", resTensor.Format());
        operandVecOut = {res4DimTensor.GetStorage()};
    }
    auto &op = functionPtr->AddOperation(Opcode::OP_CONV, operandVecIn, operandVecOut);
    SetTensorOpAttr(op, operandVecIn[INPUT_FMAP_IDX], operandVecIn[INPUT_WEIGHT_IDX], operandVecOut[0], convAttrParam);
    if (convAttrParam.isConv1D) {
        auto &reshapeResOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, operandVecOut, {resTensor.GetStorage()});
        reshapeResOp.SetAttribute("isConv", true);
    }
    return resTensor;
}

void SetConvAttrParam(const Operation &op, ConvAttrParam &convAttrParam)
{
    convAttrParam.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    convAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorIntAttribute(CONV_PADDINGS_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_PAD_ATTR_DEFAULT_LIST;
    convAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorIntAttribute(CONV_STRIDES_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorIntAttribute(CONV_DILATIONS_ATTR) :
        convAttrParam.isConv3D ? CONV3D_ATTR_DEFAULT_LIST : CONV2D_ATTR_DEFAULT_LIST;
    convAttrParam.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    convAttrParam.hasBias = (op.HasAttr(CONV_BIAS_ATTR)) ? op.GetBoolAttribute(CONV_BIAS_ATTR) : false;
    convAttrParam.isInOutTensorNZ = false;
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_FMAP_SHAPE_ATTR))
        << "Conv ori fmapshape should be set when InOut Tensor NZ mode.";
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_WEIGHT_SHAPE_ATTR))
        << "Conv ori weightshape should be set when InOut Tensor NZ mode.";
    convAttrParam.oriFmapShape = op.GetVectorIntAttribute(CONV_ORI_FMAP_SHAPE_ATTR);
    convAttrParam.oriWeightShape = op.GetVectorIntAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR);
    convAttrParam.oriResShape = op.GetVectorIntAttribute(CONV_ORI_RES_SHAPE_ATTR);
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &cTensorPtr,
                         const ConvAttrParam &convAttrParam, ConvGraphNodes &tensorGraphNodes)
{
    // set tensor GraphNodes
    size_t operandVecSize = SHAPE_DIM2 + static_cast<size_t>(convAttrParam.hasBias);
    ASSERT(ConvExpandFuncError::EXPANDFUNC_PARAMS_INVALID,  operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: "
        << "Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", Conv Common Input: " << SHAPE_DIM2 << ", hasBias: " << convAttrParam.hasBias;

    tensorGraphNodes.fmapTensorPtr = operandVec[INPUT_FMAP_IDX];
    tensorGraphNodes.weightTensorPtr = operandVec[INPUT_WEIGHT_IDX];
    if (convAttrParam.hasBias) {
        tensorGraphNodes.biasTensorPtr = operandVec[INPUT_BIAS_IDX];
    }
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR,
        tensorGraphNodes.fmapTensorPtr != nullptr && tensorGraphNodes.weightTensorPtr != nullptr)
        << "Expected aTensorPtr and bTensorPtr to be non-nullptr.";

    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, cTensorPtr != nullptr)
        << "cTensorPtr is nullptr.";
    tensorGraphNodes.resTensorPtr = cTensorPtr;
}

void SetConvShapeInfo(const TileShape &tileShape, const ConvGraphNodes &tensorGraphNodes,
                      const ConvAttrParam &convAttrParam, ConvTileInfo &convTileInfo)
{
    // set org shape
    convTileInfo.orgBatch = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_N_IDX] : convAttrParam.oriFmapShape[NCHW_N_IDX];
    convTileInfo.orgHin = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_H_IDX] : convAttrParam.oriFmapShape[NCHW_H_IDX];
    convTileInfo.orgWin = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_W_IDX] : convAttrParam.oriFmapShape[NCHW_W_IDX];
    convTileInfo.orgCin = convAttrParam.isConv3D ?
        convAttrParam.oriFmapShape[NCDHW_C_IDX] : convAttrParam.oriFmapShape[NCHW_C_IDX];
    convTileInfo.orgHout = convAttrParam.isConv3D ?
        convAttrParam.oriResShape[NCDHW_H_IDX] : convAttrParam.oriResShape[NCHW_H_IDX];
    convTileInfo.orgWout = convAttrParam.isConv3D ?
        convAttrParam.oriResShape[NCDHW_W_IDX] : convAttrParam.oriResShape[NCHW_W_IDX];
    convTileInfo.orgDin = convAttrParam.isConv3D ? convAttrParam.oriFmapShape[NCDHW_D_IDX] : 1;
    convTileInfo.orgDout = convAttrParam.isConv3D ? convAttrParam.oriResShape[NCDHW_D_IDX] : 1;
    convTileInfo.cin0 = ALIGN_SIZE_32 / BytesOf(tensorGraphNodes.fmapTensorPtr->Datatype());
    convTileInfo.orgCout = convAttrParam.isConv3D ?
        convAttrParam.oriWeightShape[NCDHW_N_IDX] : convAttrParam.oriWeightShape[NCHW_N_IDX];
    convTileInfo.orgKh = convAttrParam.isConv3D ?
        convAttrParam.oriWeightShape[NCDHW_H_IDX] : convAttrParam.oriWeightShape[NCHW_H_IDX];
    convTileInfo.orgKw = convAttrParam.isConv3D ?
        convAttrParam.oriWeightShape[NCDHW_W_IDX] : convAttrParam.oriWeightShape[NCHW_W_IDX];
    convTileInfo.orgKd = convAttrParam.isConv3D ? convAttrParam.oriWeightShape[NCDHW_D_IDX] : 1;
    int64_t cinPerGroup = convTileInfo.orgCin / convAttrParam.groups;
    convTileInfo.orgHoutWout = convTileInfo.orgHout * convTileInfo.orgWout;
    convTileInfo.kPerGroup = ConvAlignB(cinPerGroup, convTileInfo.cin0) *
        convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.coutPerGroup = convTileInfo.orgCout / convAttrParam.groups;
    // set tileshape info
    auto &convTile = tileShape.GetConvTile();
    convTileInfo.kAL1 = convTile.tileL1Info.tileCinFmap * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.kBL1 = convTile.tileL1Info.tileCinWeight * convTileInfo.orgKh * convTileInfo.orgKw;
    convTileInfo.nBL1 = convTile.tileL1Info.tileN;
    convTileInfo.hAL1In = convTile.tileL1Info.tileHin;
    convTileInfo.wAL1In = convTile.tileL1Info.tileWin;
    convTileInfo.hAL1Out = convTile.tileL1Info.tileHout;
    convTileInfo.wAL1Out = convTile.tileL1Info.tileWout;
    convTileInfo.kL0 = convTile.tileL0Info.tileK;
    convTileInfo.hL0 = convTile.tileL0Info.tileH;
    convTileInfo.wL0 = convTile.tileL0Info.tileW;
    convTileInfo.nL0 = convTile.tileL0Info.tileN;
}

LogicalTensorPtr ConstructBiasTile(Function &function, const ConvGraphNodes &tensorGraphNodes, ConvIterInfo &iterInfo,
                                   ConvTileInfo &convTileInfo)
{
    std::vector<int64_t> dstBiasL1Shape = std::vector<int64_t>{1, iterInfo.nL0Size};
    std::vector<int64_t> dstBiasL1Offset = std::vector<int64_t>{0,
        iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.nL1Offset + iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasl1TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.biasTensorPtr->Datatype(),
                                        dstBiasL1Shape, SymbolicScalar::FromConcrete(dstBiasL1Shape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasL1Tensor", NodeType::LOCAL);
    dstBiasl1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBiasL1Shape));
    auto &viewOpBiasL1 = function.AddOperation(Opcode::OP_VIEW, {tensorGraphNodes.biasTensorPtr},
                                               {dstBiasl1TensorPtr});
    auto viewAttributeBiasL1 = std::make_shared<ViewOpAttribute>(dstBiasL1Offset, MemoryType::MEM_L1,
            SymbolicScalar::FromConcrete(dstBiasL1Offset), dstBiasl1TensorPtr->GetDynValidShape());
    viewOpBiasL1.SetOpAttribute(viewAttributeBiasL1);
    viewOpBiasL1.SetAttribute(Matrix::A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(Matrix::CopyInMode::ND2ND));

    std::vector<int64_t> dstBiasBtShape = std::vector<int64_t>{1, iterInfo.nL0Size};
    std::vector<int64_t> dstBiasBtOffset = std::vector<int64_t>{0, iterInfo.nL0Offset};
    LogicalTensorPtr dstBiasBtTensorPtr =
        std::make_shared<LogicalTensor>(function, DataType::DT_FP32, dstBiasBtShape,
                                        SymbolicScalar::FromConcrete(dstBiasBtShape),
                                        tensorGraphNodes.biasTensorPtr->Format(), "biasBtTensor", NodeType::LOCAL);
    dstBiasBtTensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBiasBtShape));
    auto &viewOpBiasBt = function.AddOperation(Opcode::OP_VIEW, {dstBiasl1TensorPtr}, {dstBiasBtTensorPtr});
    auto viewAttributeBiasBt = std::make_shared<ViewOpAttribute>(dstBiasBtOffset, MemoryType::MEM_BT,
            SymbolicScalar::FromConcrete(dstBiasBtOffset), dstBiasBtTensorPtr->GetDynValidShape());
    viewOpBiasBt.SetOpAttribute(viewAttributeBiasBt);

    return dstBiasBtTensorPtr;
}

void SetImg2ColAttr(Operation &load3dOpAl0, const ConvAttrParam &convAttrParam, ConvIterInfo &iterInfo,
                    const ConvTileInfo &convTileInfo)
{
    int64_t strideH = convAttrParam.strides[0];
    int64_t strideW = convAttrParam.strides[1];
    int64_t dilationH = convAttrParam.dilations[0];
    int64_t dilationW = convAttrParam.dilations[1];
    int64_t dilatedKernelH = (convTileInfo.orgKh - 1) * dilationH + 1;
    int64_t dilatedKernelW = (convTileInfo.orgKw - 1) * dilationW + 1;
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::strideH, strideH);
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::strideW, strideW);
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::dilationH, dilationH);
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::dilationW, dilationW);
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::filterH, convTileInfo.orgKh);
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::filterW, convTileInfo.orgKw);
    // cal H padding
    if (iterInfo.hL1InOffset >= 0) {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingTop, 0);
    } else {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingTop, 0 - iterInfo.hL1InOffset);
    }
    int64_t hinAL1Used = (iterInfo.houtL1Size - 1) * strideH + dilatedKernelH;
    int64_t hinBottomPadOffset = iterInfo.hL1InOffset + hinAL1Used;
    if (hinBottomPadOffset > convTileInfo.orgHin) {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingBottom,
                                 hinBottomPadOffset - convTileInfo.orgHin);
    } else {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingBottom, 0);
    }
    // cal W padding
    if (iterInfo.wL1InOffset >= 0) {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingLeft, 0);
    } else {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingLeft, 0 - iterInfo.wL1InOffset);
    }
    int64_t winAL1Used = (iterInfo.woutL1Size - 1) * strideW + dilatedKernelW;
    int64_t winRightPadOffset = iterInfo.wL1InOffset + winAL1Used;
    if (winRightPadOffset > convTileInfo.orgWin) {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingRight,
                                 winRightPadOffset - convTileInfo.orgWin);
    } else {
        load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::paddingRight, 0);
    }
    // cal postm postk
    int64_t mStartPt = iterInfo.hL0Offset * iterInfo.woutL1Size + iterInfo.wL0Offset;
    int64_t kStartPt = iterInfo.kL0Offset % convTileInfo.kAL1;
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::postM, mStartPt);
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::postK, kStartPt);
    // set pad value
    load3dOpAl0.SetAttribute(L12L0ConvOpAttributeKey::padValue, 0);
    // set conv/conv3d flag
    load3dOpAl0.SetAttribute("isConv", true);
    load3dOpAl0.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
}

void SetCopyInAL1Op(Operation &copyInOpAl1, const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                    const ConvAttrParam &convAttrParam, const std::vector<int64_t> &dstAL1Shape,
                    const std::vector<int64_t> &srcGmValidShape, const int64_t &srcCinOffset)
{
    copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::isFmap, true);
    copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
    copyInOpAl1.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode, static_cast<int64_t>(CopyInMode::COPY_MOD_DN2NZ));
    copyInOpAl1.SetAttribute("src_d_stride", convAttrParam.isConv3D ? convAttrParam.dilations[2] : 1);
    int64_t src_n_offset = iterInfo.batchOffset;
    int64_t src_c_offset = iterInfo.groupOffset * (convTileInfo.orgCin / convAttrParam.groups) + srcCinOffset;
    int64_t src_d_offset = convAttrParam.isConv3D ?
        (iterInfo.dinL1Offset + (iterInfo.kL0Offset / convTileInfo.kPerGroup) * convAttrParam.dilations[2]) : 0;
    int64_t src_h_offset = iterInfo.hL1InOffset > 0 ? iterInfo.hL1InOffset : 0;
    int64_t src_w_offset = iterInfo.wL1InOffset > 0 ? iterInfo.wL1InOffset : 0;
    std::vector<int64_t> srcFmapGmOffset = {src_n_offset, src_c_offset, src_h_offset, src_w_offset};
    if (convAttrParam.isConv3D) {
        srcFmapGmOffset = {src_n_offset, src_c_offset, src_d_offset, src_h_offset, src_w_offset};
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcFmapGmOffset),
        MemoryType::MEM_L1, OpImmediate::Specified(srcGmValidShape), OpImmediate::Specified(dstAL1Shape),
        OpImmediate::Specified(dstAL1Shape)
    );
    copyInOpAl1.SetOpAttribute(copyAttr);
    copyInOpAl1.SetAttribute("l1_tile_shape", SymbolicScalar::FromConcrete(dstAL1Shape));
    iterInfo.aL1UpadateFlag = false;
}

LogicalTensorPtr ConstructFmapTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                   const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                                   LogicalTensorPtr &dstAL1TensorPtr, const ConvAttrParam &convAttrParam)
{
    if (iterInfo.kL0Offset % convTileInfo.kAL1 == 0) {
        iterInfo.aL1UpadateFlag = true;
    }
    // L1层级 Fmap 展开
    if (iterInfo.aL1UpadateFlag) {
        iterInfo.kAL1Size =
            std::min((convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset), convTileInfo.kAL1);
        int64_t cin1AL1Size = (iterInfo.kAL1Size / convTileInfo.cin0) / (convTileInfo.orgKh * convTileInfo.orgKw);
        std::vector<int64_t> dstAL1Shape = std::vector<int64_t>{1, cin1AL1Size,
            iterInfo.hinL1Size, iterInfo.winL1Size, convTileInfo.cin0};
        int64_t srcCinOffset =
            (iterInfo.kL0Offset % convTileInfo.kPerGroup) / (convTileInfo.orgKh * convTileInfo.orgKw);
        int64_t srcGmCin = std::min(convTileInfo.orgCin / convAttrParam.groups - srcCinOffset,
                                    convTileInfo.kAL1 / (convTileInfo.orgKh * convTileInfo.orgKw));
        std::vector<int64_t> srcGmValidShape =
            std::vector<int64_t>{1, srcGmCin, iterInfo.hinL1Size, iterInfo.winL1Size};
        if (convAttrParam.isConv3D) {
            iterInfo.dkAL1Size = 1;
            if (iterInfo.kAL1Size > convTileInfo.kPerGroup) {
                srcCinOffset = 0;
                iterInfo.dkAL1Size = iterInfo.kAL1Size / convTileInfo.kPerGroup;
                cin1AL1Size = (iterInfo.kAL1Size / (iterInfo.dkAL1Size * convTileInfo.cin0)) /
                    (convTileInfo.orgKh * convTileInfo.orgKw);
            }
            dstAL1Shape = std::vector<int64_t>{1, iterInfo.dkAL1Size, cin1AL1Size,
                                               iterInfo.hinL1Size, iterInfo.winL1Size, convTileInfo.cin0};
            srcGmValidShape =
                std::vector<int64_t>{1, srcGmCin, iterInfo.dkAL1Size, iterInfo.hinL1Size, iterInfo.winL1Size};
        }
        dstAL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(),
            dstAL1Shape, SymbolicScalar::FromConcrete(dstAL1Shape), tensorGraphNodes.fmapTensorPtr->Format(),
            "aL1Tensor", NodeType::LOCAL);
        dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL1Shape));
        auto &copyInOpAl1 = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV, {tensorGraphNodes.fmapTensorPtr},
                                                  {dstAL1TensorPtr});
        copyInOpAl1.SetAttribute("isConv", true);
        SetCopyInAL1Op(copyInOpAl1, convTileInfo, iterInfo, convAttrParam, dstAL1Shape, srcGmValidShape, srcCinOffset);
    }
    // 二层展开
    // load3dv2()
    std::vector<int64_t> dstAL0Shape =
        std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE), iterInfo.kL0Size};
    LogicalTensorPtr dstAL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstAL0Shape,
                                        SymbolicScalar::FromConcrete({iterInfo.mL0Size, iterInfo.kL0Size}),
                                        tensorGraphNodes.fmapTensorPtr->Format(), "aL0Tensor", NodeType::LOCAL);
    dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete({iterInfo.mL0Size, iterInfo.kL0Size}));
    auto &load3dOpAl0 = function.AddOperation(Opcode::OP_LOAD3D_CONV, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    load3dOpAl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstAL0Shape));
    SetImg2ColAttr(load3dOpAl0, convAttrParam, iterInfo, convTileInfo);
    return dstAL0TensorPtr;
}

void SetCopyInBL1Op(Operation &copyInOpBl1, const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                    const ConvAttrParam &convAttrParam, const std::vector<int64_t> &dstBL1Shape,
                    const std::vector<int64_t> &srcGmValidShape, const int64_t &srcCinOffset)
{
    copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::isFmap, false);
    copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
    copyInOpBl1.SetAttribute(LoadStoreConvOpAttributeKey::copyInMode, static_cast<int64_t>(CopyInMode::COPY_MOD_DN2NZ));
    int64_t src_n_offset = iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.nL1Offset;
    int64_t src_c_offset = srcCinOffset;
    int64_t src_d_offset = 0;
    if (convAttrParam.isConv3D) {
        src_d_offset = (iterInfo.doL1Offset * convAttrParam.strides[2] - convAttrParam.paddings[4]) < 0 ?
            (convTileInfo.orgKd - iterInfo.dkBL1SrcOffset + (iterInfo.kL0Offset / convTileInfo.kPerGroup)) :
            (iterInfo.kL0Offset / convTileInfo.kPerGroup);
    }
    int64_t src_h_offset = 0;
    int64_t src_w_offset = 0;
    std::vector<int64_t> srcWeightGmOffset = {src_n_offset, src_c_offset, src_h_offset, src_w_offset};
    if (convAttrParam.isConv3D) {
        srcWeightGmOffset = {src_n_offset, src_c_offset, src_d_offset, src_h_offset, src_w_offset};
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcWeightGmOffset),
        MemoryType::MEM_L1, OpImmediate::Specified(srcGmValidShape), OpImmediate::Specified(dstBL1Shape),
        OpImmediate::Specified(dstBL1Shape)
    );
    copyInOpBl1.SetOpAttribute(copyAttr);
    copyInOpBl1.SetAttribute("l1_tile_shape", SymbolicScalar::FromConcrete(dstBL1Shape));
    iterInfo.bL1UpadateFlag = false;
}

LogicalTensorPtr ConstructWeightTile(Function &function, const ConvGraphNodes &tensorGraphNodes,
                                     const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo,
                                     LogicalTensorPtr &dstBL1TensorPtr, const ConvAttrParam &convAttrParam)
{
    if (iterInfo.kL0Offset % convTileInfo.kBL1 == 0) {
        iterInfo.bL1UpadateFlag = true;
    }
    // L1层级 Weight 展开
    if (iterInfo.bL1UpadateFlag) {
        iterInfo.kBL1Size =
            std::min(convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset, convTileInfo.kBL1);
        std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{iterInfo.kBL1Size / convTileInfo.cin0,
            CeilDiv(iterInfo.nL1Size, MKN_N_VALUE), MKN_N_VALUE, convTileInfo.cin0};
        int64_t srcCinOffset =
            (iterInfo.kL0Offset % convTileInfo.kPerGroup) / (convTileInfo.orgKh * convTileInfo.orgKw);
        int64_t srcGmCin = std::min(convTileInfo.orgCin / convAttrParam.groups - srcCinOffset,
                                    convTileInfo.kBL1 / (convTileInfo.orgKh * convTileInfo.orgKw));
        std::vector<int64_t> srcGmValidShape =
            std::vector<int64_t>{iterInfo.nL1Size, srcGmCin, convTileInfo.orgKh, convTileInfo.orgKw};
        if (convAttrParam.isConv3D) {
            iterInfo.dkBL1Size = 1;
            if (iterInfo.kBL1Size > convTileInfo.kPerGroup) {
                srcCinOffset = 0;
                iterInfo.dkBL1Size = iterInfo.kBL1Size / convTileInfo.kPerGroup;
            }
            dstBL1Shape = std::vector<int64_t>{iterInfo.kBL1Size / convTileInfo.cin0,
                                               CeilDiv(iterInfo.nL1Size, MKN_N_VALUE), MKN_N_VALUE, convTileInfo.cin0};
            srcGmValidShape = std::vector<int64_t>{iterInfo.nL1Size, srcGmCin,
                                                   iterInfo.dkBL1Size, convTileInfo.orgKh, convTileInfo.orgKw};
        }
        dstBL1TensorPtr =
            std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL1Shape,
                                            SymbolicScalar::FromConcrete(dstBL1Shape),
                                            tensorGraphNodes.weightTensorPtr->Format(), "bL1Tensor", NodeType::LOCAL);
        dstBL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL1Shape));
        auto &copyInOpBl1 = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV, {tensorGraphNodes.weightTensorPtr},
                                                  {dstBL1TensorPtr});
        copyInOpBl1.SetAttribute("isConv", true);
        SetCopyInBL1Op(copyInOpBl1, convTileInfo, iterInfo, convAttrParam, dstBL1Shape, srcGmValidShape, srcCinOffset);
    }
    // load2d()
    std::vector<int64_t> dstBL0Shape =
        std::vector<int64_t>{iterInfo.kL0Size, ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
    LogicalTensorPtr dstBL0TensorPtr =
        std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL0Shape,
                                        SymbolicScalar::FromConcrete({iterInfo.kL0Size, iterInfo.nL0Size}),
                                        tensorGraphNodes.weightTensorPtr->Format(), "bL0Tensor", NodeType::LOCAL);
    dstBL0TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete({iterInfo.kL0Size, iterInfo.nL0Size}));
    auto &load2dOpBl0 = function.AddOperation(Opcode::OP_LOAD2D_CONV, {dstBL1TensorPtr}, {dstBL0TensorPtr});
    load2dOpBl0.SetAttribute(L12L0ConvOpAttributeKey::postK, iterInfo.kL0Offset % convTileInfo.kBL1);
    load2dOpBl0.SetAttribute(L12L0ConvOpAttributeKey::postN, iterInfo.nL0Offset);
    load2dOpBl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstBL0Shape));
    load2dOpBl0.SetAttribute("isConv", true);
    return dstBL0TensorPtr;
}

void SetAMulBAttr(const ConvGraphNodes &tensorGraphNodes, const ConvTileInfo &convTileInfo, Operation &op)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_OP_NULLPTR,
        tensorGraphNodes.fmapTensorPtr != nullptr && tensorGraphNodes.weightTensorPtr != nullptr &&
        tensorGraphNodes.resTensorPtr != nullptr)
        << "Expected fmapTensorPtr, weightTensorPtr, and resTensorPtr to be non-nullptr.";

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.fmapTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.weightTensorPtr->Format()) << 1) |
                     (static_cast<int64_t>(tensorGraphNodes.resTensorPtr->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(A_MUL_B_ACT_M, convTileInfo.hL0 * convTileInfo.wL0);
    op.SetAttribute(A_MUL_B_ACT_K, convTileInfo.kL0);
    op.SetAttribute(A_MUL_B_ACT_N, convTileInfo.nL0);

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
    }
}

LogicalTensorPtr DoMmad(Function &function, const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
                        ConvGraphNodes &tileGraphNodes, const ConvTileInfo &convTileInfo, const ConvIterInfo &iterInfo)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR,
        tileGraphNodes.fmapTensorPtr != nullptr && tileGraphNodes.weightTensorPtr != nullptr &&
        tileGraphNodes.resTensorPtr != nullptr) << "Inputs and res must be non-nullptr.";
    // MMAD node add
    std::vector<LogicalTensorPtr> mmadInputs;
    std::vector<LogicalTensorPtr> mmadOutputs;
    const std::string MmadOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr};
        if (convAttrParam.hasBias) {
            ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tileGraphNodes.biasTensorPtr != nullptr)
                << "bias must be non-nullptr when hasBias Flag.";
            mmadInputs.push_back(tileGraphNodes.biasTensorPtr);
        }
    } else {
        mmadInputs = {tileGraphNodes.fmapTensorPtr, tileGraphNodes.weightTensorPtr, tileGraphNodes.cL0PartialSumPtr};
    }

    if (iterInfo.isLastK) {
        mmadOutputs = {tileGraphNodes.resTensorPtr};
    } else {
        std::vector<int64_t> cL0PartialSumShape =
            {ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE), ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
        tileGraphNodes.cL0PartialSumPtr =
            std::make_shared<LogicalTensor>(function, DataType::DT_FP32, cL0PartialSumShape,
                                            SymbolicScalar::FromConcrete({iterInfo.mL0Size, iterInfo.nL0Size}),
                                            TileOpFormat::TILEOP_NZ, "cL0PartialSumTensor", NodeType::LOCAL);
        tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape({iterInfo.mL0Size, iterInfo.nL0Size});
        mmadOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto &aMulBOp = function.AddOperation(MmadOpStr, mmadInputs, mmadOutputs);
    aMulBOp.SetAttribute("isConv", true);

    SetAMulBAttr(tensorGraphNodes, convTileInfo, aMulBOp);

    return mmadOutputs[0];
}

void ConstrucCopyOutTile(Function &function, const ConvAttrParam &convAttrParam,
                         const ConvGraphNodes &tensorGraphNodes, const ConvTileInfo &convTileInfo,
                         const ConvIterInfo &iterInfo, const LogicalTensorPtr &resCl0TensorPtr)
{
    auto &fixpipeOpRes = function.AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {resCl0TensorPtr},
                                               {tensorGraphNodes.resTensorPtr});
    fixpipeOpRes.SetAttribute("isConv", true);
    // set fixpipe copy out validshape
    fixpipeOpRes.SetAttribute(LoadStoreConvOpAttributeKey::copyOutMode,
                              static_cast<int64_t>(CopyOutMode::COPY_MOD_NZ2DN));
    fixpipeOpRes.SetAttribute(LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);
    fixpipeOpRes.SetAttribute("res_tile_shape",
                                SymbolicScalar::FromConcrete(tensorGraphNodes.resTensorPtr->shape));
    int64_t dst_n_offset = iterInfo.batchOffset;
    int64_t dst_c_offset =
        iterInfo.groupOffset * convTileInfo.coutPerGroup + iterInfo.nL1Offset + iterInfo.nL0Offset;
    int64_t dst_d_offset = iterInfo.doL1Offset;
    int64_t dst_h_offset = iterInfo.hL1OutOffset + iterInfo.hL0Offset;
    int64_t dst_w_offset = iterInfo.wL1OutOffset + iterInfo.wL0Offset;
    std::vector<int64_t> dstResGmOffset = {dst_n_offset, dst_c_offset, dst_h_offset, dst_w_offset};
    if (convAttrParam.isConv3D) {
        dstResGmOffset = {dst_n_offset, dst_c_offset, dst_d_offset, dst_h_offset, dst_w_offset};
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L1,
        OpImmediate::Specified(dstResGmOffset),
        OpImmediate::Specified({iterInfo.mL0Size, iterInfo.nL0Size}),
        OpImmediate::Specified({iterInfo.mL0Size, iterInfo.nL0Size}),
        OpImmediate::Specified({iterInfo.mL0Size, iterInfo.nL0Size})
    );
    fixpipeOpRes.SetOpAttribute(copyAttr);
}

void Cal3DDkL1Size(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo, const ConvAttrParam &convAttrParam)
{
    // cal dk in L1, not support dk in L1 = 0 now, kerneld <= padd
    iterInfo.dkL1Size = 1;
    if (convAttrParam.isConv3D) {
        iterInfo.dkL1Size = convTileInfo.orgKd;
        iterInfo.dinL1Offset = iterInfo.doL1Offset * convAttrParam.strides[2] - convAttrParam.paddings[4];
        int64_t srcDkOffset = iterInfo.dinL1Offset;
        if (iterInfo.dinL1Offset < 0) {
            int64_t tmpKd = CeilDiv(-iterInfo.dinL1Offset, convAttrParam.dilations[2]);
            iterInfo.dkL1Size -= tmpKd;
            iterInfo.dkBL1SrcOffset = iterInfo.dkL1Size;
            srcDkOffset = iterInfo.dinL1Offset + tmpKd * convAttrParam.dilations[2];
        }
        int64_t kdL1EndOffset = iterInfo.dinL1Offset + (convTileInfo.orgKd - 1) * convAttrParam.dilations[2] + 1;
        if (kdL1EndOffset > convTileInfo.orgDin) {
            int64_t tmpKd = CeilDiv(kdL1EndOffset - convTileInfo.orgDin, convAttrParam.dilations[2]);
            iterInfo.dkL1Size -= tmpKd;
        }
        iterInfo.dinL1Offset = srcDkOffset;
    }
}

void UpdateL1IterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo, const ConvAttrParam &convAttrParam)
{
    // update iterInfo L1
    // cal winL1Size
    iterInfo.houtL1Size = std::min(convTileInfo.orgHout - iterInfo.hL1OutOffset, convTileInfo.hAL1Out);
    iterInfo.hL1InOffset = iterInfo.hL1OutOffset * convAttrParam.strides[0] - convAttrParam.paddings[0];
    int64_t needHL1Size = (iterInfo.houtL1Size - 1) * convAttrParam.strides[0] +
        (convTileInfo.orgKh - 1) * convAttrParam.dilations[0] + 1;
    if (iterInfo.hL1InOffset < 0) {
        // start pos locate in pad
        iterInfo.hinL1Size = needHL1Size + iterInfo.hL1InOffset;
        if (iterInfo.hL1InOffset + needHL1Size <= 0) {
            // all locate in pad
            iterInfo.hinL1Size = 0;
        }
        if (iterInfo.hinL1Size > convTileInfo.orgHin) {
            // w all load l1
            iterInfo.hinL1Size = convTileInfo.orgHin;
        }
    } else if (convTileInfo.orgHin - iterInfo.hL1InOffset <= 0){
        // start pos locate in bottom pad
        iterInfo.hinL1Size = 0;
    } else {
        iterInfo.hinL1Size = std::min(convTileInfo.orgHin - iterInfo.hL1InOffset, needHL1Size);
    }
    // cal winL1Size
    iterInfo.woutL1Size = std::min(convTileInfo.orgWout - iterInfo.wL1OutOffset, convTileInfo.wAL1Out);
    iterInfo.wL1InOffset = iterInfo.wL1OutOffset * convAttrParam.strides[1] - convAttrParam.paddings[2];
    int64_t needWL1Size = (iterInfo.woutL1Size - 1) * convAttrParam.strides[1] +
        (convTileInfo.orgKw - 1) * convAttrParam.dilations[1] + 1;
    if (iterInfo.wL1InOffset < 0) {
        // start pos locate in pad
        iterInfo.winL1Size = needWL1Size + iterInfo.wL1InOffset;
        if (iterInfo.wL1InOffset + needWL1Size <= 0) {
            // all locate in pad
            iterInfo.winL1Size = 0;
        }
        if (iterInfo.winL1Size > convTileInfo.orgWin) {
            // w all load l1
            iterInfo.winL1Size = convTileInfo.orgWin;
        }
    } else if (convTileInfo.orgWin - iterInfo.wL1InOffset <= 0){
        // start pos locate in right pad
        iterInfo.winL1Size = 0;
    } else {
        iterInfo.winL1Size = std::min(convTileInfo.orgWin - iterInfo.wL1InOffset, needWL1Size);
    }
    // cal nL1Size
    iterInfo.nL1Size = std::min(convTileInfo.coutPerGroup - iterInfo.nL1Offset, convTileInfo.nBL1);
    Cal3DDkL1Size(convTileInfo, iterInfo, convAttrParam);
}

void UpdateL0IterInfo(const ConvTileInfo &convTileInfo, ConvIterInfo &iterInfo)
{
    // update iterInfo
    iterInfo.kL0Size = std::min(convTileInfo.kPerGroup * iterInfo.dkL1Size - iterInfo.kL0Offset, convTileInfo.kL0);
    iterInfo.isFirstK = iterInfo.kL0Offset == 0 ? true : false;
    iterInfo.isLastK =
        iterInfo.kL0Offset + convTileInfo.kL0 >= convTileInfo.kPerGroup * iterInfo.dkL1Size ? true : false;
}

void IterL0ExpandFunc(Function &function, ConvIterInfo &iterInfo, ConvTileInfo &convTileInfo,
                      const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
                      ConvGraphNodes &tileGraphNodes)
{
    LogicalTensorPtr fmapL1TensorPtr = nullptr;
    LogicalTensorPtr weightL1TensorPtr = nullptr;
    LogicalTensorPtr resCl0TensorPtr = nullptr;
    for (iterInfo.nL0Offset = 0; iterInfo.nL0Offset < iterInfo.nL1Size; iterInfo.nL0Offset += convTileInfo.nL0) {
        iterInfo.nL0Size = std::min(iterInfo.nL1Size - iterInfo.nL0Offset, convTileInfo.nL0);
        for (iterInfo.hL0Offset = 0; iterInfo.hL0Offset < iterInfo.houtL1Size;
             iterInfo.hL0Offset += convTileInfo.hL0) {
            for (iterInfo.wL0Offset = 0; iterInfo.wL0Offset < iterInfo.woutL1Size;
                 iterInfo.wL0Offset += convTileInfo.wL0) {
                if (convTileInfo.wL0 == convTileInfo.wAL1Out) {
                    iterInfo.mL0Size =
                        std::min(iterInfo.houtL1Size * iterInfo.woutL1Size - iterInfo.hL0Offset * iterInfo.woutL1Size,
                                 convTileInfo.hL0 * convTileInfo.wL0);
                } else {
                    iterInfo.mL0Size = std::min(iterInfo.woutL1Size - iterInfo.wL0Offset, convTileInfo.wL0);
                }
                // bias 载入
                if (convAttrParam.hasBias) {
                    // get bias in bt tile for mmad
                    tileGraphNodes.biasTensorPtr =
                        ConstructBiasTile(function, tensorGraphNodes, iterInfo, convTileInfo);
                }
                // set res tile
                std::vector<int64_t> dstCL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, MKN_M_VALUE),
                                                                        ConvAlignB(iterInfo.nL0Size, MKN_N_VALUE)};
                tileGraphNodes.resTensorPtr =
                    std::make_shared<LogicalTensor>(function, tensorGraphNodes.fmapTensorPtr->Datatype(), dstCL0Shape,
                                                    SymbolicScalar::FromConcrete(dstCL0Shape),
                                                    tensorGraphNodes.fmapTensorPtr->Format(), "cL0Tensor",
                                                    NodeType::LOCAL);
                for (iterInfo.kL0Offset = 0; iterInfo.kL0Offset < convTileInfo.kPerGroup * iterInfo.dkL1Size;
                     iterInfo.kL0Offset += convTileInfo.kL0) {
                    UpdateL0IterInfo(convTileInfo, iterInfo);
                    // fmap and weight link
                    tileGraphNodes.fmapTensorPtr = ConstructFmapTile(function, tensorGraphNodes, convTileInfo,
                                                                     iterInfo, fmapL1TensorPtr, convAttrParam);
                    tileGraphNodes.weightTensorPtr = ConstructWeightTile(function, tensorGraphNodes, convTileInfo,
                                                                         iterInfo, weightL1TensorPtr, convAttrParam);
                    // add mmad node
                    resCl0TensorPtr = DoMmad(function, convAttrParam, tensorGraphNodes, tileGraphNodes,
                                             convTileInfo, iterInfo);
                }
                ConstrucCopyOutTile(function, convAttrParam, tensorGraphNodes, convTileInfo, iterInfo, resCl0TensorPtr);
            }
        }
    }
}

void IterOneBatchFunc(Function &function, ConvIterInfo &iterInfo, ConvTileInfo &convTileInfo,
                      const ConvAttrParam &convAttrParam, const ConvGraphNodes &tensorGraphNodes,
                      ConvGraphNodes &tileGraphNodes)
{
    for (iterInfo.doL1Offset = 0; iterInfo.doL1Offset < convTileInfo.orgDout; iterInfo.doL1Offset +=1) {
        for (iterInfo.nL1Offset = 0; iterInfo.nL1Offset < convTileInfo.coutPerGroup;
            iterInfo.nL1Offset += convTileInfo.nBL1) {
            iterInfo.bL1UpadateFlag = true;
            for (iterInfo.hL1OutOffset = 0; iterInfo.hL1OutOffset < convTileInfo.orgHout;
                iterInfo.hL1OutOffset += convTileInfo.hAL1Out) {
                for (iterInfo.wL1OutOffset = 0; iterInfo.wL1OutOffset < convTileInfo.orgWout;
                    iterInfo.wL1OutOffset += convTileInfo.wAL1Out) {
                    iterInfo.aL1UpadateFlag = true;
                    UpdateL1IterInfo(convTileInfo, iterInfo, convAttrParam);
                    // iterate L0 buffer expand
                    IterL0ExpandFunc(function, iterInfo, convTileInfo, convAttrParam,
                                     tensorGraphNodes, tileGraphNodes);
                }
            }
        }
    }
}

void ConstructTileGraph(Function &function, const TileShape &tileShape,
                        const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &cTensorPtr,
                        const Operation &op)
{
    // op attr set
    ConvAttrParam convAttrParam;
    SetConvAttrParam(op, convAttrParam);
    // set tensor graph node info
    ConvGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, convAttrParam, tensorGraphNodes);
    // save tile info
    ConvTileInfo convTileInfo;
    SetConvShapeInfo(tileShape, tensorGraphNodes, convAttrParam, convTileInfo);
    // save iter info
    ConvIterInfo iterInfo;
    // set tile graph node info
    ConvGraphNodes tileGraphNodes;

    for (iterInfo.groupOffset = 0; iterInfo.groupOffset < convAttrParam.groups; iterInfo.groupOffset += 1) {
        for (iterInfo.batchOffset = 0; iterInfo.batchOffset < convTileInfo.orgBatch; iterInfo.batchOffset += 1) {
            IterOneBatchFunc(function, iterInfo, convTileInfo, convAttrParam, tensorGraphNodes, tileGraphNodes);
        }
    }
}
Tensor Conv(DataType outType, const Tensor &inputTensor, const Tensor &weightTensor, const std::vector<int64_t> &strides,
            const std::vector<int64_t> &paddings, const std::vector<int64_t> &dilations, const ConvExtendParam &extendParam,
            const int64_t groups)
{
    std::vector<int64_t> finalPaddings = paddings;
    std::vector<int64_t> finalDilations = dilations;
    std::vector<int64_t> finalStrides = strides;
    if (dilations.size() == CONV3D_INPUT_DIM - 2 && strides.size() == CONV3D_INPUT_DIM - 2 && paddings.size() == 2 * (CONV3D_INPUT_DIM - 2)) {
        finalDilations = rotateVector(dilations, 1);
        finalStrides = rotateVector(strides, 1);
        finalPaddings = rotateVector(paddings, 2);
    }
    const Tensor& biasTensor = extendParam.biasTensor;
    // init and set attr
    ConvAttrParam convAttrParam(finalPaddings, finalStrides, finalDilations, groups);
    CheckConvOperands(outType, inputTensor, weightTensor, biasTensor, convAttrParam);
    int64_t batchOut = inputTensor.GetShape()[NCHW_N_IDX];
    int64_t cOut = weightTensor.GetShape()[NCHW_N_IDX];
    int64_t hOut = ConvComputeHo(inputTensor, weightTensor, convAttrParam);
    int64_t wOut = ConvComputeWo(inputTensor, weightTensor, convAttrParam);
    std::vector<int64_t> resTensorShape{batchOut, cOut, hOut, wOut};
    if (convAttrParam.isConv1D) {
        resTensorShape = {batchOut, cOut, wOut};
        convAttrParam.paddings.insert(convAttrParam.paddings.begin(), 2, 0);
        convAttrParam.strides.insert(convAttrParam.strides.begin(), 1);
        convAttrParam.dilations.insert(convAttrParam.dilations.begin(), 1);
    }
    if (convAttrParam.isConv3D) {
        int64_t dOut = ConvComputeDo(inputTensor, weightTensor, convAttrParam);
        resTensorShape = {batchOut, cOut, dOut, hOut, wOut};
    }
    Tensor resTensor(outType, resTensorShape, "TensorC");
    resTensor.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(resTensorShape));
    return ConstructTensorGraph(inputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

} //namespace Conv
}
}