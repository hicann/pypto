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
 * \file cube_operation_impl.cpp
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
#include "interface/utils/matmul_error.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "tilefwk/tile_shape.h"

namespace npu {
namespace tile_fwk {
namespace Matrix {
const float EPSILON = 1e-6f;

template <typename T>
auto CeilAlign(T num_1, T num_2) -> T
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

inline bool CheckValidShape(const LogicalTensorPtr &tensorPtr)
{
    if (tensorPtr == nullptr) {
        return false;
    }
    return tensorPtr->GetDynValidShape().size() == SHAPE_DIM2;
}

inline size_t GetAlignSize(DataType dataType)
{
    bool isB4 = dataType == DataType::DT_FP4_E2M1X2 || dataType == DataType::DT_FP4_E1M2X2;
    return isB4 ? ALIGN_SIZE_64 : ALIGN_SIZE_32;
}

template <typename T1, typename T2 = T1>
LogicalTensorPtr AddOpView(Function &function, const LogicalTensorPtr &srcTensorPtr,
    const MatmulTensorInfo &dstTensorInfo, const std::map<std::string, T1> opAttr = {},
    const std::map<std::string, T2> extraOpAttr = {}) {
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, srcTensorPtr != nullptr)
        << "Original tensor for OpView operation is nullptr.";
    auto dstShape = dstTensorInfo.shape;
    if (dstTensorInfo.transFlag) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, dstShape.size() == SHAPE_DIM2 || dstShape.size() == SHAPE_DIM3)
            << "destination shape dimension is invalid: Expected dimensions == " << SHAPE_DIM2 << " or " << SHAPE_DIM3
            << ", actual dimensions: " << dstShape.size();
        std::swap(dstShape[0], dstShape[1]);
    }
    LogicalTensorPtr dstTensorPtr = std::make_shared<LogicalTensor>(function, dstTensorInfo.dtype, dstShape,
        SymbolicScalar::FromConcrete(dstShape), dstTensorInfo.format, dstTensorInfo.name, dstTensorInfo.nodeType);
    dstTensorPtr->UpdateDynValidShape(
        GetViewValidShape(srcTensorPtr->GetDynValidShape(), dstTensorInfo.offset, {}, dstTensorInfo.shape));
    if (dstTensorInfo.transFlag) {
        auto &dstValidShape = dstTensorPtr->GetDynValidShape();
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
            dstValidShape.size() == SHAPE_DIM2 || dstValidShape.size() == SHAPE_DIM3)
            << "dstValidShape dimension is invalid: Expected dimensions == " << SHAPE_DIM2 << " or " << SHAPE_DIM3
            << ", actual dimensions: " << dstValidShape.size();
        std::swap(dstValidShape[0], dstValidShape[1]);
    }
    auto &viewOp = function.AddOperation(Opcode::OP_VIEW, {srcTensorPtr}, {dstTensorPtr});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        dstTensorInfo.offset, SymbolicScalar::FromConcrete(dstTensorInfo.offset), dstTensorPtr->GetDynValidShape());
    viewAttribute->SetToType(dstTensorInfo.memType);
    viewOp.SetOpAttribute(viewAttribute);
    for (const auto &attrPair : opAttr) {
        viewOp.SetAttribute(attrPair.first, attrPair.second);
    }
    for (const auto &attrPair : extraOpAttr) {
        viewOp.SetAttribute(attrPair.first, attrPair.second);
    }
    return dstTensorPtr;
}

LogicalTensorPtr AddOpView(Function &function, const LogicalTensorPtr &srcTensorPtr,
                           const MatmulTensorInfo &dstTensorInfo)
{
    return AddOpView<int64_t>(function, srcTensorPtr, dstTensorInfo);
}

void SetAMulBAttr(const MatmulGraphNodes &tensorGraphNodes, const MatmulAttrParam &attrParam, Operation &op) {
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.aTensorPtr != nullptr)
        << "aTensorPtr is nullptr, check input tensor A.";

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.bTensorPtr != nullptr)
        << "bTensorPtr is nullptr, check input tensor B.";

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.outTensorPtr != nullptr)
        << "outTensorPtr is nullptr, check output tensor C.";

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.aTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.bTensorPtr->Format()) << 1) |
                     // 2含义：cTensorPtr的索引，同时也是cTensor NZ信息的编码偏移位数
                     (static_cast<int64_t>(tensorGraphNodes.outTensorPtr->Format()) << 2);
    op.SetAttribute(MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(A_MUL_B_ACT_M, attrParam.mValue);
    op.SetAttribute(A_MUL_B_ACT_K, attrParam.kValue);
    op.SetAttribute(A_MUL_B_ACT_N, attrParam.nValue);
    op.SetAttribute(A_MUL_B_GM_ACC, attrParam.gmAccumulationFlag);
    op.SetAttribute(A_MUL_B_TRANS_MODE_ATTR, static_cast<int64_t>(attrParam.transMode));

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
        op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(attrParam.reluType));
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, attrParam.scaleValue));
    }
}

void SetTensorGraphAttr(
    Operation &op, const MatmulExtendParam &param, bool gmAccumulationFlag, const MatmulAttrParam &attrParam)
{
    op.SetAttribute(A_MUL_B_GM_ACC, gmAccumulationFlag);
    op.SetAttribute(A_MUL_B_TRANS_A, attrParam.transA);
    op.SetAttribute(A_MUL_B_TRANS_B, attrParam.transB);
    op.SetAttribute(A_MUL_B_BIAS_ATTR, (param.biasTensor.GetStorage() != nullptr));
    op.SetAttribute(A_MUL_B_RELU_ATTR, static_cast<int64_t>(param.reluType));
    op.SetAttribute(A_MUL_B_TRANS_MODE_ATTR, static_cast<int64_t>(param.transMode));
    // means perchannel
    if (param.scaleTensor.GetStorage() != nullptr) {
        op.SetAttribute(A_MUL_B_VECTOR_QUANT_FLAG, true);
    }
    // means pertensor
    if (fabs(param.scaleValue - 0) > EPSILON) {
        uint32_t scaleValueTmp = 0;
        memcpy_s(&scaleValueTmp, sizeof(scaleValueTmp), &param.scaleValue, sizeof(param.scaleValue));
        op.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, static_cast<uint64_t>(scaleValueTmp)));
    }
    // mx matmul
    if (attrParam.hasMXScale) {
        op.SetAttribute(A_MUL_B_MX_ATTR, true);
        op.SetAttribute(A_MUL_B_SCALE_A_COPY_IN_MODE,
            attrParam.transAScale ? static_cast<int64_t>(CopyInMode::DN2NZ) : static_cast<int64_t>(CopyInMode::ND2NZ));
        op.SetAttribute(A_MUL_B_SCALE_B_COPY_IN_MODE,
            attrParam.transBScale ? static_cast<int64_t>(CopyInMode::DN2NZ) : static_cast<int64_t>(CopyInMode::ND2NZ));
    }

    auto matrixSize = TileShape::Current().GetMatrixSize();
    if (matrixSize.size() < MATRIX_MAXSIZE) {
        op.SetAttribute(A_MUL_B_ACT_M, 0);
        op.SetAttribute(A_MUL_B_ACT_N, 0);
        op.SetAttribute(A_MUL_B_ACT_K, 0);
        return;
    }
    op.SetAttribute(A_MUL_B_ACT_M, matrixSize[M_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_N, matrixSize[N_INDEX]);
    op.SetAttribute(A_MUL_B_ACT_K, matrixSize[K_INDEX]);
}

void SetMatmulAttrParam(const Operation &op, MatmulAttrParam &param)
{
    param.mValue = (op.HasAttr(A_MUL_B_ACT_M)) ? op.GetIntAttribute(A_MUL_B_ACT_M) : 0;
    param.kValue = (op.HasAttr(A_MUL_B_ACT_K)) ? op.GetIntAttribute(A_MUL_B_ACT_K) : 0;
    param.nValue = (op.HasAttr(A_MUL_B_ACT_N)) ? op.GetIntAttribute(A_MUL_B_ACT_N) : 0;
    param.reluType = (op.HasAttr(A_MUL_B_RELU_ATTR)) ? op.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    param.scaleValue = (op.HasAttr(A_MUL_B_SCALE_ATTR)) ? op.GetElementAttribute(A_MUL_B_SCALE_ATTR).GetUnsignedData()
                                                        : Element(DataType::DT_UINT64, 0).GetUnsignedData();
    param.hasBias = (op.HasAttr(A_MUL_B_BIAS_ATTR)) ? op.GetBoolAttribute(A_MUL_B_BIAS_ATTR) : false;
    param.hasScale = (op.HasAttr(A_MUL_B_VECTOR_QUANT_FLAG)) ? op.GetBoolAttribute(A_MUL_B_VECTOR_QUANT_FLAG) : false;
    param.hasMXScale = op.HasAttr(A_MUL_B_MX_ATTR);
    param.transA = (op.HasAttr(A_MUL_B_TRANS_A)) ? op.GetBoolAttribute(A_MUL_B_TRANS_A) : false;
    param.transB = (op.HasAttr(A_MUL_B_TRANS_B)) ? op.GetBoolAttribute(A_MUL_B_TRANS_B) : false;
    param.gmAccumulationFlag = (op.HasAttr(A_MUL_B_GM_ACC)) ? op.GetBoolAttribute(A_MUL_B_GM_ACC) : false;
    param.transMode = (op.HasAttr(A_MUL_B_TRANS_MODE_ATTR)) ? op.GetIntAttribute(A_MUL_B_TRANS_MODE_ATTR) : 0;
    if (param.hasMXScale) {
        param.transAScale = op.GetIntAttribute(A_MUL_B_SCALE_A_COPY_IN_MODE) == static_cast<int64_t>(CopyInMode::DN2NZ);
        param.transBScale = op.GetIntAttribute(A_MUL_B_SCALE_B_COPY_IN_MODE) == static_cast<int64_t>(CopyInMode::DN2NZ);
    }
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr> &operandVec, const LogicalTensorPtr &cTensorPtr,
    const MatmulAttrParam &param, MatmulGraphNodes &tensorGraphNodes) {
    size_t mxScaleSize = static_cast<size_t>(param.hasMXScale) * SHAPE_DIM2;
    size_t operandVecSize =
        SHAPE_DIM2 + static_cast<size_t>(param.hasScale + param.hasBias + param.gmAccumulationFlag) + mxScaleSize;
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", SHAPE_DIM2: " << SHAPE_DIM2 << ", hasScale: " << param.hasScale << ", hasBias: " << param.hasBias
        << ", gmAccumulationFlag: " << param.gmAccumulationFlag << ", hasMXScale: " << param.hasMXScale;
    tensorGraphNodes.aTensorPtr = operandVec[0];
    tensorGraphNodes.bTensorPtr = operandVec[1];
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.aTensorPtr != nullptr)
        << "aTensorPtr is nullptr, check input tensor A.";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.bTensorPtr != nullptr)
        << "bTensorPtr is nullptr, check input tensor B.";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, cTensorPtr != nullptr) << "cTensorPtr is nullptr.";
    tensorGraphNodes.outTensorPtr = cTensorPtr;
    size_t extraDim = static_cast<size_t>(param.hasScale) | (static_cast<size_t>(param.hasBias) << 1) |
                      (static_cast<size_t>(param.gmAccumulationFlag) << 2) |
                      (static_cast<size_t>(param.hasMXScale) << 3); // 2、3含义：编码偏移
    switch (extraDim) {
        case 0: // 无bias，无scale, 无gmTensor
            break;
        case 1: // 有scale
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 2: // 2含义：有bias
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 3: // 3含义：有bias, 有scale
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.scaleTensorPtr = operandVec[SHAPE_DIM3];
            break;
        case 4: // 4含义：有gmTensor
            tensorGraphNodes.gmAccumulationTensorPtr = operandVec[SHAPE_DIM2];
            break;
        case 8: // 8含义: mxmatmul场景
            tensorGraphNodes.aScaleTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.bScaleTensorPtr = operandVec[SHAPE_DIM3];
            break;
        case 10: // 10含义: mxmatmul场景，有bias
            tensorGraphNodes.aScaleTensorPtr = operandVec[SHAPE_DIM2];
            tensorGraphNodes.bScaleTensorPtr = operandVec[SHAPE_DIM3];
            tensorGraphNodes.biasTensorPtr = operandVec[SHAPE_DIM4];
            break;
        default: ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, false) << "Invalid tensor graph";
    }
}

Status CheckOperandShape(const Tensor &operand1, const Tensor &operand2) {
    // 检查 shape 维度一致性
    const Shape shape1 = operand1.GetShape();
    const Shape shape2 = operand2.GetShape();
    size_t operand1Dim = shape1.size();
    size_t operand2Dim = shape2.size();
    size_t offsetSize1 = operand1.GetStorage()->offset.size();
    size_t offsetSize2 = operand2.GetStorage()->offset.size();

    // 检查输入维度的一致性
    const bool isDimSame = (operand1Dim == operand2Dim);
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, isDimSame)
        << "Shape dimension mismatch: operand1=" << operand1Dim << ", operand2=" << operand2Dim;

    // 检查 shape 与 offset 的一致性
    const bool isOperand1OffsetMatch = (operand1Dim == offsetSize1);
    const bool isOperand2OffsetMatch = (operand2Dim == offsetSize2);
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, isOperand1OffsetMatch)
        << "operand1 shape size(" << operand1Dim << ") != offset size(" << offsetSize1 << ")";
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, isOperand2OffsetMatch)
        << "operand2 shape size(" << operand2Dim << ") != offset size(" << offsetSize2 << ")";

    // 检查最小维度
    const bool Op1DimValid = (operand1Dim >= SHAPE_DIM2);
    const bool Op2DimValid = (operand2Dim >= SHAPE_DIM2);
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, Op1DimValid) << "operand1 dimension(" << operand1Dim << ") must be >= 2";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, Op2DimValid) << "operand2 dimension(" << operand2Dim << ") must be >= 2";

    // 检查每个维度的值
    for (size_t i = 0; i < operand1Dim; ++i) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, shape1[i] > 0)
            << "operand1 dim[" << i << "] = " << shape1[i] << ", must be > 0";
    }

    for (size_t i = 0; i < operand2Dim; ++i) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, shape2[i] > 0)
            << "operand2 dim[" << i << "] = " << shape2[i] << ", must be > 0";
    }

    MATMUL_LOGD("CheckOperandShape: PASS");
    return SUCCESS;
}

Status CheckL1L0Tile(
    const int64_t L0Tile, const int64_t L1Tile, const std::string &L0TileName, const std::string &L1TileName) {
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, L0Tile != 0) << L0TileName << " cannot be zero, got " << L0Tile;

    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, L0Tile <= L1Tile && L1Tile % L0Tile == 0)
        << "Invalid L1/L0 relation: " << L0TileName << "=" << L0Tile << ", " << L1TileName << "=" << L1Tile
        << ", require " << L0TileName << " <= " << L1TileName << " && " << L1TileName << " % " << L0TileName << " == 0";

    return SUCCESS;
}

Status CheckCubeTiling(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int32_t kBL1Idx = 2;
    const int64_t kL0 = cubeTile.k[0];
    const int64_t kL1a = cubeTile.k[1];
    const int64_t kL1b = cubeTile.k[kBL1Idx];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t mL1 = cubeTile.m[1];
    const int64_t nL0 = cubeTile.n[0];
    const int64_t nL1 = cubeTile.n[1];
    ASSERT(
        MatmulErrorCode::ERR_CONFIG_TILE, kL0 > 0 && kL1a > 0 && kL1b > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0)
        << "Invalid tile values: kL0=" << kL0 << ", kL1a=" << kL1a << ", kL1b=" << kL1b << ", mL0=" << mL0
        << ", mL1=" << mL1 << ", nL0=" << nL0 << ", nL1=" << nL1;

    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, kL0 % ALIGN_SIZE_16 == 0 && nL0 % ALIGN_SIZE_16 == 0)
        << "kL0(" << kL0 << ") and nL0(" << nL0 << ") must be aligned to 16 elements";
    // 检查 L1/L0 tile 关系
    if (CheckL1L0Tile(kL0, kL1a, "kL0", "kL1a") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(kL0, kL1b, "kL0", "kL1b") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(nL0, nL1, "nL0", "nL1") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(mL0, mL1, "mL0", "mL1") != SUCCESS) {
        return FAILED;
    }
    size_t alignSizeA = GetAlignSize(operand1.GetDataType());
    size_t alignSizeB = GetAlignSize(operand2.GetDataType());
    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, alignSizeA != 0 && alignSizeB != 0)
        << "The alignSize is zero, please check!! alignSizeA=" << alignSizeA << ", alignSizeB=" << alignSizeB;

    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
        << "kL0 * sizeof(dtype) = " << kL0 * BytesOf(operand1.GetDataType()) << " bytes, must be 32-byte aligned";

    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, nL0 * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0)
        << "nL0 * sizeof(dtype) = " << nL0 * BytesOf(operand2.GetDataType()) << " bytes, must be 32-byte aligned";
    if (operand1.Format() == TileOpFormat::TILEOP_ND) {
        if (attrParam.transA) { // For ND A transpose, mL0 must be 32B aligned
            ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, mL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0)
                << "mL0 memory not aligned when transA=true: " << mL0 * BytesOf(operand1.GetDataType())
                << " bytes, must be 32-byte aligned";
        }
    }
    return SUCCESS;
}

void CheckOperandShapeBound(const Tensor &operand) {
    auto opFormat = operand.Format();
    size_t alignSize = GetAlignSize(operand.GetDataType());
    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, alignSize != 0)
        << "The alignSize is zero, please check!!";
    if (opFormat == TileOpFormat::TILEOP_ND) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, operand.GetShape().back() <= SHAPE_INNER_AXIS_MAX_SIZE)
            << "Current inner axis: " << operand.GetShape().back()
            << ", when input is ND format, inner axis must be less than 65535";

        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
            operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] <= std::numeric_limits<int32_t>::max())
            << "Current outer axis: " << operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2]
            << ", when input is ND format, outer axis must be less than 2^31 - 1";

        if (operand.GetDataType() == DataType::DT_FP4_E2M1X2 || operand.GetDataType() == DataType::DT_FP4_E1M2X2) {
            ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, (operand.GetShape().back() & 1) == 0)
                << "Current inner axis: " << operand.GetShape().back()
                << ", when input is ND format and 4bit dtype, inner axis must be even number";
        }
    } else {
        ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT,
            operand.GetShape().back() * BytesOf(operand.GetDataType()) % ALIGN_SIZE_32 == 0)
            << "Current inner axis: " << operand.GetShape().back()
            << ", when input is NZ format, inner axis shape must be 32-byte aligned";

        ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT,
            operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] % ALIGN_SIZE_16 == 0)
            << "Current outer axis: " << operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2]
            << ", when input is NZ format, outer axis shape must be 16-element aligned";
    }
}

void CheckByteAlign(const Tensor &operand, const std::string &tileName, int64_t tileVal) {
    size_t alignSize = GetAlignSize(operand.GetDataType());
    int64_t totalBytes = tileVal * BytesOf(operand.GetDataType());
    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, alignSize != 0)
        << "The alignSize is zero, please check!!";

    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, tileVal * BytesOf(operand.GetDataType()) % alignSize == 0)
        << "Current length of " << tileName << ": " << (size_t)totalBytes
        << " bytes, the length must be aligned to 32 bytes(4bit dtype must be aligned to 64)";
}

void CheckElementAlign(const std::string &tileName, int64_t tileVal) {
    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, tileVal % ALIGN_SIZE_16 == 0)
        << "Current length of " << tileName << ": " << (size_t)tileVal
        << " elements, the length must be aligned to 16 elements";
}

void CheckNZFormatAligned(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int64_t kL0 = cubeTile.k[0];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t nL0 = cubeTile.n[0];
    auto opFormatA = operand1.Format();
    auto opFormatB = operand2.Format();
    if (opFormatA == TileOpFormat::TILEOP_NZ) {
        if (attrParam.transA) {
            CheckByteAlign(operand1, "mL0", mL0);
            CheckElementAlign("kL0", kL0);
        } else {
            CheckByteAlign(operand1, "kL0", kL0);
            CheckElementAlign("mL0", mL0);
        }
    }
    if (opFormatB == TileOpFormat::TILEOP_NZ) {
        if (attrParam.transB) {
            CheckByteAlign(operand2, "kL0", kL0);
            CheckElementAlign("nL0", nL0);
        } else {
            CheckByteAlign(operand2, "nL0", nL0);
            CheckElementAlign("kL0", kL0);
        }
    }
}

void CheckCMatrixNZFormatAligned(const DataType &outType, const Tensor &operand, const MatmulAttrParam &attrParam) {
    auto &cubeType = TileShape::Current().GetCubeTile();
    const int64_t nL0 = cubeType.n[0];
    if (attrParam.isCMatrixNZ) {
        int64_t nView = attrParam.transB ? operand.GetShape()[operand.GetShape().size() - SHAPE_DIM2] :
                                           operand.GetShape()[operand.GetShape().size() - 1];
        if (outType == DataType::DT_INT32) {
            ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, nView % ALIGN_SIZE_16 == 0)
                << "Current nView: " << nView
                << " elements, nView must be aligned to 16 elements when CMatrix is NZ and outType is int32";
            ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, nL0 % ALIGN_SIZE_16 == 0)
                << "Current nL0: " << nL0
                << " elements, nL0 must be aligned to 16 elements when CMatrix is NZ and outType is int32";
        } else {
            const bool nViewIsAlign = ((nView * BytesOf(outType)) % ALIGN_SIZE_32) == 0;
            const bool nL0IsAlign = ((nL0 * BytesOf(outType)) % ALIGN_SIZE_32) == 0;
            ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, nViewIsAlign) << "Current nView: " << nView * BytesOf(outType)
                << " bytes, nView must be aligned to 32 bytes when CMatrix is NZ";
            ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, nL0IsAlign) << "Current nL0: " << nL0 * BytesOf(outType)
                << " bytes, nL0 must be aligned to 32 bytes when CMatrix is NZ";
        }
    }
}

void CheckBiasParam(DataType inDtype, const MatmulExtendParam &param = {}) {
    if (param.biasTensor.GetStorage() == nullptr) {
        return;
    }
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, param.biasTensor.Format() == TileOpFormat::TILEOP_ND)
        << "Only support TILEOP_ND.";
    if (inDtype == DataType::DT_BF16 || inDtype == DataType::DT_FP32) {
        ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, param.biasTensor.GetDataType() == DataType::DT_FP32)
            << "When input tensor is DT_BF16 or DT_FP32, bias must be DT_FP32.";
    } else if (inDtype == DataType::DT_FP16) {
        ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH,
            param.biasTensor.GetDataType() == DataType::DT_FP32 || param.biasTensor.GetDataType() == DataType::DT_FP16)
            << "When input tensor is DT_FP16, bias must be DT_FP32 or DT_FP16.";
    } else if (inDtype == DataType::DT_INT8) {
        ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, param.biasTensor.GetDataType() == DataType::DT_INT32)
            << "When input tensor is DT_INT8, bias must be DT_INT32.";
    }
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, param.biasTensor.GetShape().size() == SHAPE_DIM2)
        << "Bias tensor shape dimension mismatch: Expected " << SHAPE_DIM2 << " dimensions, got "
        << param.biasTensor.GetShape().size();
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, param.biasTensor.GetShape()[0] == 1)
        << "Bias tensor first dimension mismatch: Expected first dimension to be 1, got "
        << param.biasTensor.GetShape()[0];
}

void CheckFixpipeParam(DataType inDtype, DataType outDtype, const MatmulExtendParam &param = {}) {
    if (param.scaleTensor.GetStorage() != nullptr) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, param.scaleTensor.Format() == TileOpFormat::TILEOP_ND)
            << "Only support TILEOP_ND.";

        ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, param.scaleTensor.GetDataType() == DataType::DT_INT64 ||
                                                        param.scaleTensor.GetDataType() == DataType::DT_UINT64)
            << "scaleTensor dataType: " << DataType2String(param.scaleTensor.GetDataType())
            << ". scaleTensor only support int64 and uint64 dtype currently.";

        ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, outDtype == DataType::DT_FP16 && inDtype == DataType::DT_INT8)
            << "Data type mismatch in fixpipe scenario. Expected inDtype to be DT_INT8 and outDtype to be DT_FP16.";

        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, param.scaleTensor.GetShape()[0] == 1)
            << "Scale tensor first dimension mismatch. Expected first dimension to be 1, got "
            << param.scaleTensor.GetShape()[0];
    }
    if (fabs(param.scaleValue - 0) > EPSILON) {
        ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, outDtype == DataType::DT_FP16 && inDtype == DataType::DT_INT8)
            << "Data type mismatch in pertensor scenario. Expected inDtype to be DT_INT8 and outDtype to be DT_FP16.";
    }
    if (inDtype == DataType::DT_INT8 && outDtype == DataType::DT_FP16) {
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
            fabs(param.scaleValue - 0) > EPSILON || param.scaleTensor.GetStorage() != nullptr)
            << "Quantization error in INT8→FP16 path: scaleValue must not be 0.0f, OR scaleTensor must not be null.";
    }
}

void CheckTransModeParam(DataType inDtype, const MatmulExtendParam &param = {}) {
    if (param.transMode != TransMode::CAST_NONE) {
        ASSERT(MatmulErrorCode::ERR_PARAM_UNSUPPORTED, inDtype == DataType::DT_FP32)
            << "The param of transMode is only supported when input data type is DT_FP32.";
    }
}

void CheckGmAccumulationParam(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix,
    const MatmulAttrParam &attrParam, const MatmulExtendParam &param = {}) {
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (!cubeTile.enableSplitK) {
        return;
    }
    ASSERT(MatmulErrorCode::ERR_CONFIG_UNSUPPORTED, !attrParam.isCMatrixNZ)
        << "Gm accumulation with output NZ format is not supported.";

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, param.scaleTensor.GetStorage() == nullptr &&
                                                   param.biasTensor.GetStorage() == nullptr &&
                                                   fabs(param.scaleValue - 0) < EPSILON)
        << "Fixpipe and bias cannot be used simultaneously with GM ACC";

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, outType == DataType::DT_FP32 || outType == DataType::DT_INT32)
        << "Output data type only support FP32 and INT32 when using GM accumulated";

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
        << "Both aMatrix and bMatrix cannot get storage";

    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, aMatrixValidShape.size() == SHAPE_DIM2 &&
                                                   bMatrixValidShape.size() == SHAPE_DIM2 &&
                                                   cubeTile.k.size() == MAX_K_DIM_SIZE)
        << "The validShapes of aMatrix and bMatrix must be 2 Dim. Additionally, the K TileShape must be 3 Dim";

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        aMatrix.GetShape().size() == SHAPE_DIM2 && bMatrix.GetShape().size() == SHAPE_DIM2)
        << "The shapes of aMatrix and bMatrix must be 2 Dim";
    int64_t kSizeA = attrParam.transA ? aMatrix.GetShape()[aMatrix.GetShape().size() - SHAPE_DIM2] :
                                        aMatrix.GetShape()[aMatrix.GetShape().size() - 1];
    int64_t kSizeB = attrParam.transB ? bMatrix.GetShape()[bMatrix.GetShape().size() - 1] :
                                        bMatrix.GetShape()[bMatrix.GetShape().size() - SHAPE_DIM2];
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kSizeA == kSizeB)
        << "Matrix K dimension mismatch, kSizeA: " << kSizeA << ", kSizeB: " << kSizeB;
}

void CheckOperandDtype(DataType outType, const Tensor &operand1, const Tensor &operand2) {
    ASSERT(MatmulErrorCode::ERR_PARAM_UNSUPPORTED, outType == DataType::DT_FP32 || outType == DataType::DT_FP16 ||
                                                       outType == DataType::DT_BF16 || outType == DataType::DT_INT32)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16, DT_INT32 are supported.";
    const DataType operand1Dtype = operand1.GetDataType();
    const DataType operand2Dtype = operand2.GetDataType();
    const bool isOperand1Fp8 = (operand1Dtype == DataType::DT_FP8E5M2 || operand1Dtype == DataType::DT_FP8E4M3);
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH,
        !isOperand1Fp8 || (operand2Dtype == DataType::DT_FP8E5M2 || operand2Dtype == DataType::DT_FP8E4M3))
        << "When operand1 is of type DT_FP8E4M3 or DT_FP8E5M2, operand2 must be DT_FP8E4M3 or DT_FP8E5M2. operand1 "
           "dataType: "
        << DataType2String(operand1Dtype) << ", operand2 dataType: " << DataType2String(operand2Dtype);

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        (isOperand1Fp8 == false) || (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510))
        << "When operand1 data type is DT_FP8E5M2, only DAV_3510 architecture is supported.";

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        operand1Dtype != DataType::DT_FP8E5M2 || operand1.Format() == TileOpFormat::TILEOP_ND)
        << "When operand1 data type is DT_FP8E5M2, format must be ND.";

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        operand2Dtype != DataType::DT_FP8E5M2 || operand2.Format() == TileOpFormat::TILEOP_ND)
        << "When operand2 data type is DT_FP8E5M2, format must be ND.";

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, isOperand1Fp8 || (operand1Dtype == operand2Dtype))
        << "input dataType must be consistent. operand1 dataType: " << DataType2String(operand1Dtype)
        << ", operand2 dataType: " << DataType2String(operand2Dtype);
}

 Status CheckMatmulOperands(DataType outType, const Tensor &operand1, const Tensor &operand2,
    const MatmulAttrParam &attrParam, const MatmulExtendParam &param = {}) {
    MATMUL_LOGD("Begin Matmul Operand Legality Check.\n");
    // dtype valid check
    CheckOperandDtype(outType, operand1, operand2);
    // shape valid check
    CheckOperandShape(operand1, operand2);
    // GM Acc valid check
    CheckGmAccumulationParam(outType, operand1, operand2, attrParam, param);
    // tile valid check
    CheckCubeTiling(operand1, operand2, attrParam);
    // shape bound valid check
    CheckOperandShapeBound(operand1);
    CheckOperandShapeBound(operand2);
    // input NZ format valid check
    CheckNZFormatAligned(operand1, operand2, attrParam);
    // output NZ format valid check
    CheckCMatrixNZFormatAligned(outType, operand2, attrParam);
    // bias and scale valid check
    CheckBiasParam(operand1.GetDataType(), param);
    CheckFixpipeParam(operand1.GetDataType(), outType, param);
    // trans mode valid check
    CheckTransModeParam(operand1.GetDataType(), param);
    MATMUL_LOGD("Finish Matmul Operand Legality Check.\n");
    return SUCCESS;
}

void CheckMXMatmulShape(const Tensor &aTensor, const Tensor &aScaleTensor, const Tensor &bTensor,
    const Tensor &bScaleTensor, const MatmulAttrParam &attrParam) {
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        aScaleTensor.GetShape().size() == SHAPE_DIM3 && bScaleTensor.GetShape().size() == SHAPE_DIM3)
        << "The dimension of scaleTensor for mxmatmul must be equal to 3! The dimension of ascaleTensor: "
        << aScaleTensor.GetShape().size() << ", The dimension of bscaleTensor: " << bScaleTensor.GetShape().size();

    int64_t mSize = attrParam.transA ? aTensor.GetShape()[1] : aTensor.GetShape()[0];
    int64_t nSize = attrParam.transB ? bTensor.GetShape()[0] : bTensor.GetShape()[1];
    int64_t kSize = attrParam.transA ? aTensor.GetShape()[0] : aTensor.GetShape()[1];

    int64_t mScaleSize = attrParam.transAScale ? aScaleTensor.GetShape()[1] : aScaleTensor.GetShape()[0];
    int64_t kAScaleSize0 = attrParam.transAScale ? aScaleTensor.GetShape()[0] : aScaleTensor.GetShape()[1];
    int64_t kAScaleSize1 = aScaleTensor.GetShape()[SHAPE_DIM2];
    int64_t kBScaleSize0 = attrParam.transBScale ? bScaleTensor.GetShape()[1] : bScaleTensor.GetShape()[0];
    int64_t kBScaleSize1 = bScaleTensor.GetShape()[SHAPE_DIM2];
    int64_t nScaleSize = attrParam.transBScale ? bScaleTensor.GetShape()[0] : bScaleTensor.GetShape()[1];

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kAScaleSize0 == kBScaleSize0)
        << "Scale Matrix K dimension mismatch, kAScaleSize: " << kAScaleSize0 << ", kBScaleSize: " << kBScaleSize0;

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, kAScaleSize1 == NUM2 && kBScaleSize1 == NUM2)
        << "Scale Matrix Inner axis must be equal to 2, AScale Inner axis: " << kAScaleSize1
        << ", BScale Inner axis: " << kBScaleSize1;

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, mSize == mScaleSize)
        << "Scale Matrix M dimension mismatch, mScaleSize: " << mScaleSize << ", mSize: " << mSize;

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, nSize == nScaleSize)
        << "Scale Matrix N dimension mismatch, nScaleSize: " << nScaleSize << ", nSize: " << nSize;

    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, kSize % ALIGN_SIZE_64 == 0)
        << "Current kSize: " << kSize << ", kSize must be aligned to 64 element when using MX Matmul";

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kAScaleSize0 == kSize / ALIGN_SIZE_64)
        << "Matrix K dimension is not a multiple of 64. Expected: ksize / 64 = " << kAScaleSize0
        << ", but got ksize / 64: " << kSize / ALIGN_SIZE_64;
}

Status CheckMXMatmulOperands(const Tensor &aTensor, const Tensor &aScaleTensor, const Tensor &bTensor,
    const Tensor &bScaleTensor, const MatmulAttrParam &attrParam) {
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH,
        aScaleTensor.GetDataType() == DataType::DT_FP8E8M0 && bScaleTensor.GetDataType() == DataType::DT_FP8E8M0)
        << "input scale dataType must be DT_FP8E8M0. aScaleTensor dataType: "
        << DataType2String(aScaleTensor.GetDataType())
        << ", bScaleTensor dataType: " << DataType2String(bScaleTensor.GetDataType());
    DataType inDType = aTensor.GetDataType();
    static const std::unordered_set<DataType> supportedTypes = {
        DataType::DT_FP8E4M3, DataType::DT_FP8E5M2, DataType::DT_FP4_E2M1X2, DataType::DT_FP4_E1M2X2};
    ASSERT(MatmulErrorCode::ERR_PARAM_UNSUPPORTED, supportedTypes.find(inDType) != supportedTypes.end())
        << "Unsupported input data type. Only support DT_FP8E4M3, DT_FP8E5M2, DT_FP4_E2M1X2, DT_FP4_E1M2X2.";
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int64_t kL0 = cubeTile.k[0];
    ASSERT(MatmulErrorCode::ERR_CONFIG_ALIGNMENT, kL0 % ALIGN_SIZE_64 == 0)
        << "Current length of kL0: " << kL0 << ", the length of kL0 for mx matmul must be aligned to 64 elements";
    CheckOperandShape(aScaleTensor, bScaleTensor);
    CheckMXMatmulShape(aTensor, aScaleTensor, bTensor, bScaleTensor, attrParam);
    return SUCCESS;
}

void SetMatmulTileInfo(const TileShape &tileShape, const MatmulAttrParam &attrParam,
    const MatmulGraphNodes &tensorGraphNodes, MatmulTileInfo &tileInfo) {
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR,
        tensorGraphNodes.aTensorPtr != nullptr && tensorGraphNodes.bTensorPtr != nullptr)
        << "Both inputs must be non-nullptr.";

    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.aTensorPtr->GetShape().size() == SHAPE_DIM2 &&
                                                   tensorGraphNodes.bTensorPtr->GetShape().size() == SHAPE_DIM2)
        << "Invalid tensor shape dimension, expected both tensors to have exactly " << SHAPE_DIM2
        << " dimensions. aTensorPtr shape dim: " << tensorGraphNodes.aTensorPtr->GetShape().size()
        << ", bTensorPtr shape dim: " << tensorGraphNodes.bTensorPtr->GetShape().size();

    tileInfo.mView = attrParam.transA ? tensorGraphNodes.aTensorPtr->shape[1] : tensorGraphNodes.aTensorPtr->shape[0];
    tileInfo.nView = attrParam.transB ? tensorGraphNodes.bTensorPtr->shape[0] : tensorGraphNodes.bTensorPtr->shape[1];
    int64_t kViewA = attrParam.transA ? tensorGraphNodes.aTensorPtr->shape[0] : tensorGraphNodes.aTensorPtr->shape[1];
    int64_t kViewB = attrParam.transB ? tensorGraphNodes.bTensorPtr->shape[1] : tensorGraphNodes.bTensorPtr->shape[0];

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kViewA == kViewB)
        << "Matrix K dimension mismatch, kViewA: " << kViewA << ", kViewB: " << kViewB;
    tileInfo.kView = kViewA;

    auto &cubeTile = tileShape.GetCubeTile();
    tileInfo.tileML0 = cubeTile.m[0];
    tileInfo.tileML1 = cubeTile.m[1];
    tileInfo.tileNL0 = cubeTile.n[0];
    tileInfo.tileNL1 = cubeTile.n[1];
    tileInfo.tileKL0 = cubeTile.k[0];
    tileInfo.tileKAL1 = cubeTile.k[1];
    tileInfo.tileKBL1 = cubeTile.k[2]; // 2含义：kBL1 tile的偏移
    int64_t tileKL1Min = std::min(tileInfo.tileKAL1, tileInfo.tileKBL1);
    int64_t tileKL1Max = std::max(tileInfo.tileKAL1, tileInfo.tileKBL1);

    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE,
        tileKL1Max >= kViewA || (tileKL1Max > 0 && tileKL1Min > 0 && tileKL1Max % tileKL1Min == 0))
        << "Invalid tileKL1 configuration: tileKL1Max: " << tileKL1Max << ", kViewA: " << kViewA
        << ", tileKL1Min: " << tileKL1Min
        << ". Must satisfy: tileKL1Max >= kViewA OR (all values > 0 and tileKL1Max is divisible by tileKL1Min).";

    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tileInfo.tileKL0 > 0 && tileKL1Min % tileInfo.tileKL0 == 0)
        << "tileKL0: " << tileInfo.tileKL0 << ", tileKL1Min: " << tileKL1Min
        << ". Must have: tileKL0 > 0 AND tileKL1Min is divisible by tileKL0.";
}

LogicalTensorPtr LinkBias(Function &function, const MatmulGraphNodes &tensorGraphNodes, const TileInfo &tileInfoL1,
                          const TileInfo &tileInfoBT)
{
    if (tensorGraphNodes.biasTensorPtr == nullptr) {
        return nullptr;
    }

    MatmulTensorInfo biasL1TensorInfo{
        "biasL1Tensor",  tensorGraphNodes.biasTensorPtr->Datatype(), tileInfoL1.shape,  tileInfoL1.offset,
        NodeType::LOCAL, tensorGraphNodes.biasTensorPtr->Format(),   MemoryType::MEM_L1};
    LogicalTensorPtr biasL1TensorPtr =
        AddOpView<int64_t>(function, tensorGraphNodes.biasTensorPtr, biasL1TensorInfo,
                           {{A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(CopyInMode::ND2ND)}});

    DataType biasBtType =
        (tensorGraphNodes.aTensorPtr->Datatype() == DataType::DT_INT8) ? DataType::DT_INT32 : DataType::DT_FP32;
    MatmulTensorInfo biasBtTensorInfo{"biasBtTensor",    biasBtType,      tileInfoBT.shape,
                                      tileInfoBT.offset, NodeType::LOCAL, biasL1TensorPtr->Format(),
                                      MemoryType::MEM_BT};
    LogicalTensorPtr biasBtTensorPtr = AddOpView(function, biasL1TensorPtr, biasBtTensorInfo);
    return biasBtTensorPtr;
}

LogicalTensorPtr LinkScale(Function &function, const MatmulGraphNodes &tensorGraphNodes, const TileInfo &tileInfoL1,
                           const TileInfo &tileInfoFB)
{
    if (tensorGraphNodes.scaleTensorPtr == nullptr) {
        return nullptr;
    }

    MatmulTensorInfo scaleL1TensorInfo{
        "scaleL1Tensor", tensorGraphNodes.scaleTensorPtr->Datatype(), tileInfoL1.shape,  tileInfoL1.offset,
        NodeType::LOCAL, tensorGraphNodes.scaleTensorPtr->Format(),   MemoryType::MEM_L1};
    LogicalTensorPtr scaleL1TensorPtr =
        AddOpView<int64_t>(function, tensorGraphNodes.scaleTensorPtr, scaleL1TensorInfo,
                           {{A_MUL_B_COPY_IN_MODE, static_cast<int64_t>(CopyInMode::ND2ND)}});

    MatmulTensorInfo scaleFbTensorInfo{"scaleFbTensor",
                                       scaleL1TensorPtr->Datatype(),
                                       tileInfoFB.shape,
                                       tileInfoFB.offset,
                                       NodeType::LOCAL,
                                       scaleL1TensorPtr->Format(),
                                       MemoryType::MEM_FIX_QUANT_PRE};
    LogicalTensorPtr scaleFbTensorPtr = AddOpView(function, scaleL1TensorPtr, scaleFbTensorInfo);
    return scaleFbTensorPtr;
}

LogicalTensorPtr LinkTensorA(Function &function, const MatmulGraphNodes &tensorGraphNodes,
                             const MatmulAttrParam &attrParam, const MatmulTileInfo &tileInfo,
                             const MatmulIterInfo &iterInfo, LogicalTensorPtr &aL1TensorPtr)
{
    if (iterInfo.kOffset % tileInfo.tileKAL1 == 0) {
        std::vector<int64_t> aL1Shape = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kAL1Size, iterInfo.mL0Size}
                                                           : std::vector<int64_t>{iterInfo.mL0Size, iterInfo.kAL1Size};
        std::vector<int64_t> aL1Offset = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kOffset, iterInfo.mOffset}
                                                            : std::vector<int64_t>{iterInfo.mOffset, iterInfo.kOffset};
        MatmulTensorInfo aL1TensorInfo{"aL1Tensor", tensorGraphNodes.aTensorPtr->Datatype(), aL1Shape, aL1Offset,
            NodeType::LOCAL, tensorGraphNodes.aTensorPtr->Format(), MemoryType::MEM_L1};
        int64_t paddingMode = 0;
        // 根据是否使用了MXScale标志来决定Padding模式
        if (attrParam.hasMXScale) {
            // 如果启用了MXScale且需要进行转置，则使用外轴填充模式(PADDING_OUTER)，
            // 否则使用内轴填充模式(PADDING_INNER)
            paddingMode = attrParam.transA ? static_cast<int64_t>(PaddingMode::PADDING_OUTER) :
                                             static_cast<int64_t>(PaddingMode::PADDING_INNER);
        }
        aL1TensorPtr = AddOpView<int64_t>(function, tensorGraphNodes.aTensorPtr, aL1TensorInfo,
                                          {{COPY_IN_L1_PADDING_MODE, paddingMode}, {REMAIN_REDUNDANT_OP_FLAG, 1}});
    }
    std::vector<int64_t> aL0Shape = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kL0Size, iterInfo.mL0Size}
                                                       : std::vector<int64_t>{iterInfo.mL0Size, iterInfo.kL0Size};
    std::vector<int64_t> aL0Offset = (attrParam.transA) ? std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKAL1, 0}
                                                        : std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKAL1};
    MatmulTensorInfo aL0TensorInfo{"aL0Tensor",
                                   tensorGraphNodes.aTensorPtr->Datatype(),
                                   aL0Shape,
                                   aL0Offset,
                                   NodeType::LOCAL,
                                   tensorGraphNodes.aTensorPtr->Format(),
                                   MemoryType::MEM_L0A,
                                   attrParam.transA};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(aL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(aL0Shape);
    LogicalTensorPtr aL0TensorPtr = AddOpView<bool, std::vector<SymbolicScalar>>(function, aL1TensorPtr, aL0TensorInfo,
        {{L1_TO_L0_TRANSPOSE, attrParam.transA}}, {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return aL0TensorPtr;
}

LogicalTensorPtr LinkTensorB(Function &function, const MatmulGraphNodes &tensorGraphNodes,
                             const MatmulAttrParam &attrParam, const MatmulTileInfo &tileInfo,
                             const MatmulIterInfo &iterInfo, LogicalTensorPtr &bL1TensorPtr)
{
    if (iterInfo.kOffset % tileInfo.tileKBL1 == 0) {
        std::vector<int64_t> bL1Shape = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nL0Size, iterInfo.kBL1Size}
                                                           : std::vector<int64_t>{iterInfo.kBL1Size, iterInfo.nL0Size};
        std::vector<int64_t> bL1Offset = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nOffset, iterInfo.kOffset}
                                                            : std::vector<int64_t>{iterInfo.kOffset, iterInfo.nOffset};
        MatmulTensorInfo bL1TensorInfo{"bL1Tensor", tensorGraphNodes.bTensorPtr->Datatype(), bL1Shape, bL1Offset,
            NodeType::LOCAL, tensorGraphNodes.bTensorPtr->Format(), MemoryType::MEM_L1};
        int64_t paddingMode = 0;
        // 根据是否使用了MXScale标志来决定Padding模式
        if (attrParam.hasMXScale) {
            // 如果启用了MXScale且需要进行转置，则使用内轴填充模式(PADDING_INNER)，
            // 否则使用外轴填充模式(PADDING_OUTER)
            paddingMode = attrParam.transB ? static_cast<int64_t>(PaddingMode::PADDING_INNER) :
                                             static_cast<int64_t>(PaddingMode::PADDING_OUTER);
        }
        bL1TensorPtr = AddOpView<int64_t>(function, tensorGraphNodes.bTensorPtr, bL1TensorInfo,
                                       {{COPY_IN_L1_PADDING_MODE, paddingMode}, {REMAIN_REDUNDANT_OP_FLAG, 1}});
    }
    std::vector<int64_t> bL0Shape = (attrParam.transB) ? std::vector<int64_t>{iterInfo.nL0Size, iterInfo.kL0Size}
                                                       : std::vector<int64_t>{iterInfo.kL0Size, iterInfo.nL0Size};
    std::vector<int64_t> bL0Offset = (attrParam.transB) ? std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKBL1}
                                                        : std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKBL1, 0};
    MatmulTensorInfo bL0TensorInfo{"bL0Tensor",
                                   tensorGraphNodes.bTensorPtr->Datatype(),
                                   bL0Shape,
                                   bL0Offset,
                                   NodeType::LOCAL,
                                   tensorGraphNodes.bTensorPtr->Format(),
                                   MemoryType::MEM_L0B,
                                   attrParam.transB};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(bL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(bL0Shape);
    LogicalTensorPtr bL0TensorPtr = AddOpView<bool, std::vector<SymbolicScalar>>(function, bL1TensorPtr, bL0TensorInfo,
        {{L1_TO_L0_TRANSPOSE, attrParam.transB}}, {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return bL0TensorPtr;
}

LogicalTensorPtr LinkTensorAScale(Function &function, const MatmulGraphNodes &tensorGraphNodes,
    const MatmulAttrParam &attrParam, const MatmulTileInfo &tileInfo, const MatmulIterInfo &iterInfo,
    LogicalTensorPtr &aScaleL1TensorPtr) {
    int64_t ALIGN_64 = 64;
    int64_t tileKAScaleL1Size = CeilAlign(iterInfo.kAL1Size, ALIGN_64) / ALIGN_64;
    int64_t tileKAScaleL0Size = CeilAlign(iterInfo.kL0Size, ALIGN_64) / ALIGN_64;
    int64_t kAScaleOffset = iterInfo.kOffset / ALIGN_64;
    int64_t copyInMode =
        attrParam.transAScale ? static_cast<int64_t>(CopyInMode::DN2NZ) : static_cast<int64_t>(CopyInMode::ND2NZ);
    if (iterInfo.kOffset % tileInfo.tileKAL1 == 0) {
        std::vector<int64_t> aScaleL1Shape = attrParam.transAScale
                                                 ? std::vector<int64_t>{tileKAScaleL1Size, iterInfo.mL0Size, NUM2}
                                                 : std::vector<int64_t>{iterInfo.mL0Size, tileKAScaleL1Size, NUM2};
        std::vector<int64_t> aScaleL1Offset = attrParam.transAScale
                                                  ? std::vector<int64_t>{kAScaleOffset, iterInfo.mOffset, 0}
                                                  : std::vector<int64_t>{iterInfo.mOffset, kAScaleOffset, 0};
        MatmulTensorInfo aScaleL1TensorInfo{
            "aScaleL1Tensor", tensorGraphNodes.aScaleTensorPtr->Datatype(), aScaleL1Shape, aScaleL1Offset,
            NodeType::LOCAL, tensorGraphNodes.aScaleTensorPtr->Format(), MemoryType::MEM_L1, attrParam.transAScale};
        aScaleL1TensorPtr = AddOpView<int64_t>(function, tensorGraphNodes.aScaleTensorPtr, aScaleL1TensorInfo,
                                               {{A_MUL_B_COPY_IN_MODE, copyInMode}});
    }
    std::vector<int64_t> aScaleL0Shape = std::vector<int64_t>{iterInfo.mL0Size, tileKAScaleL0Size, NUM2};
    std::vector<int64_t> aScaleL0Offset =
        std::vector<int64_t>{0, iterInfo.kOffset % tileInfo.tileKAL1 / ALIGN_SIZE_64, 0};
    MatmulTensorInfo aScaleL0TensorInfo{"aScaleL0Tensor", tensorGraphNodes.aScaleTensorPtr->Datatype(), aScaleL0Shape,
        aScaleL0Offset, NodeType::LOCAL, tensorGraphNodes.aScaleTensorPtr->Format(), MemoryType::MEM_L0AMX};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(aScaleL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(aScaleL0Shape);
    LogicalTensorPtr aScaleL0TensorPtr =
        AddOpView<std::vector<SymbolicScalar>>(function, aScaleL1TensorPtr, aScaleL0TensorInfo,
            {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return aScaleL0TensorPtr;
}

LogicalTensorPtr LinkTensorBScale(Function &function, const MatmulGraphNodes &tensorGraphNodes,
    const MatmulAttrParam &attrParam, const MatmulTileInfo &tileInfo, const MatmulIterInfo &iterInfo,
    LogicalTensorPtr &bScaleL1TensorPtr) {
    int64_t ALIGN_64 = 64;
    int64_t tileKBScaleL1Size = CeilAlign(iterInfo.kBL1Size, ALIGN_64) / ALIGN_64;
    int64_t tileKBScaleL0Size = CeilAlign(iterInfo.kL0Size, ALIGN_64) / ALIGN_64;
    int64_t kBScaleOffset = iterInfo.kOffset / ALIGN_64;
    int64_t copyInMode =
        attrParam.transBScale ? static_cast<int64_t>(CopyInMode::DN2NZ) : static_cast<int64_t>(CopyInMode::ND2NZ);
    if (iterInfo.kOffset % tileInfo.tileKBL1 == 0) {
        std::vector<int64_t> bScaleL1Shape = attrParam.transBScale
                                                 ? std::vector<int64_t>{iterInfo.nL0Size, tileKBScaleL1Size, NUM2}
                                                 : std::vector<int64_t>{tileKBScaleL1Size, iterInfo.nL0Size, NUM2};
        std::vector<int64_t> bScaleL1Offset = attrParam.transBScale
                                                  ? std::vector<int64_t>{iterInfo.nOffset, kBScaleOffset, 0}
                                                  : std::vector<int64_t>{kBScaleOffset, iterInfo.nOffset, 0};
        MatmulTensorInfo bScaleL1TensorInfo{
            "bScaleL1Tensor", tensorGraphNodes.bScaleTensorPtr->Datatype(), bScaleL1Shape, bScaleL1Offset,
            NodeType::LOCAL, tensorGraphNodes.bScaleTensorPtr->Format(), MemoryType::MEM_L1, attrParam.transBScale};
        bScaleL1TensorPtr = AddOpView<int64_t>(function, tensorGraphNodes.bScaleTensorPtr, bScaleL1TensorInfo,
                                               {{A_MUL_B_COPY_IN_MODE, copyInMode}});
    }
    std::vector<int64_t> bScaleL0Shape = std::vector<int64_t>{tileKBScaleL0Size, iterInfo.nL0Size, NUM2};
    std::vector<int64_t> bScaleL0Offset =
        std::vector<int64_t>{iterInfo.kOffset % tileInfo.tileKBL1 / ALIGN_SIZE_64, 0, 0};
    MatmulTensorInfo bScaleL0TensorInfo{"bScaleL0Tensor", tensorGraphNodes.bScaleTensorPtr->Datatype(), bScaleL0Shape,
        bScaleL0Offset, NodeType::LOCAL, tensorGraphNodes.bScaleTensorPtr->Format(), MemoryType::MEM_L0BMX};
    std::vector<SymbolicScalar> l1ToL0Offset = SymbolicScalar::FromConcrete(bScaleL0Offset);
    std::vector<SymbolicScalar> l1ToL0Tile = SymbolicScalar::FromConcrete(bScaleL0Shape);
    LogicalTensorPtr bScaleL0TensorPtr =
        AddOpView<std::vector<SymbolicScalar>>(function, bScaleL1TensorPtr, bScaleL0TensorInfo,
            {{L1_TO_L0_OFFSET, l1ToL0Offset}, {L1_TO_L0_TILE, l1ToL0Tile}});
    return bScaleL0TensorPtr;
}

void LinkAMulB(Function &function, const MatmulGraphNodes &tensorGraphNodes, const MatmulAttrParam &attrParam,
    const MatmulIterInfo &iterInfo, MatmulGraphNodes &tileGraphNodes) {
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tileGraphNodes.aTensorPtr != nullptr &&
                                                     tileGraphNodes.bTensorPtr != nullptr &&
                                                     tileGraphNodes.outTensorPtr != nullptr)
        << "Inputs must be non-nullptr.";
    std::vector<LogicalTensorPtr> aMulBInputs;
    std::vector<LogicalTensorPtr> aMulBOutputs;
    const std::string matmulOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr};
    } else {
        aMulBInputs = {tileGraphNodes.aTensorPtr, tileGraphNodes.bTensorPtr, tileGraphNodes.cL0PartialSumPtr};
    }
    // MX matmul 场景
    if (attrParam.hasMXScale) {
        aMulBInputs.push_back(tileGraphNodes.aScaleTensorPtr);
        aMulBInputs.push_back(tileGraphNodes.bScaleTensorPtr);
    }
    if (attrParam.gmAccumulationFlag) {
        // GM 累加场景
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.gmAccumulationTensorPtr != nullptr &&
                                                       attrParam.hasBias == false && attrParam.hasScale == false)
            << "In GM accumulation mode, neither bias nor scale is allowed.";
        tileGraphNodes.gmAccumulationTensorPtr = tensorGraphNodes.gmAccumulationTensorPtr->View(
            function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
        aMulBInputs.push_back(tileGraphNodes.gmAccumulationTensorPtr);
    } else {
        // 普通场景
        if (iterInfo.isFirstK) {
            if (tileGraphNodes.biasTensorPtr != nullptr) {
                aMulBInputs.push_back(tileGraphNodes.biasTensorPtr);
            }
            if (tileGraphNodes.scaleTensorPtr != nullptr) {
                aMulBInputs.push_back(tileGraphNodes.scaleTensorPtr);
            }
        }
    }
    if (iterInfo.isLastK) {
        aMulBOutputs = {tileGraphNodes.outTensorPtr};
    } else {
        tileGraphNodes.cL0PartialSumPtr = std::make_shared<LogicalTensor>(
            function, tileGraphNodes.outTensorPtr->Datatype(), tileGraphNodes.outTensorPtr->GetShape());
        if (CheckValidShape(tileGraphNodes.aTensorPtr) && CheckValidShape(tileGraphNodes.bTensorPtr)) {
            tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(
                {tileGraphNodes.aTensorPtr->GetDynValidShape()[0], tileGraphNodes.bTensorPtr->GetDynValidShape()[1]});
        }
        aMulBOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto &aMulBOp = function.AddOperation(matmulOpStr, aMulBInputs, aMulBOutputs);
    SetAMulBAttr(tensorGraphNodes, attrParam, aMulBOp);
}

void UpdateIterInfo(const MatmulTileInfo &tileInfo, MatmulIterInfo &iterInfo) {
    iterInfo.kAL1Size = std::min(tileInfo.tileKAL1, tileInfo.kView - iterInfo.kOffset);
    iterInfo.kBL1Size = std::min(tileInfo.tileKBL1, tileInfo.kView - iterInfo.kOffset);
    iterInfo.kL0Size = std::min(tileInfo.tileKL0, tileInfo.kView - iterInfo.kOffset);
    iterInfo.isFirstK = (iterInfo.kOffset == 0);
    iterInfo.isLastK = (iterInfo.kOffset + tileInfo.tileKL0 >= tileInfo.kView);

    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tileInfo.tileKAL1 > 0 && tileInfo.tileKBL1 > 0)
        << "Both tileKAL1 and tileKBL1 must be positive: tileKAL1: " << tileInfo.tileKAL1
        << ", tileKBL1: " << tileInfo.tileKBL1;
}

void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op)
{
    MATMUL_LOGD("ConstructTileGraph: Start.");
    MatmulAttrParam attrParam;
    SetMatmulAttrParam(op, attrParam);
    // tensor graph中的数据节点
    MatmulGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, attrParam, tensorGraphNodes);
    MatmulTileInfo tileInfo;
    SetMatmulTileInfo(tileShape, attrParam, tensorGraphNodes, tileInfo);
    MatmulIterInfo iterInfo;
    // tile graph中的数据节点
    MatmulGraphNodes tileGraphNodes;

    for (iterInfo.nOffset = 0; iterInfo.nOffset < tileInfo.nView; iterInfo.nOffset += tileInfo.tileNL0) {
        iterInfo.nL0Size = std::min(tileInfo.nView - iterInfo.nOffset, tileInfo.tileNL0);
        tileGraphNodes.biasTensorPtr =
            LinkBias(function, tensorGraphNodes, TileInfo({{1, iterInfo.nL0Size}, {0, iterInfo.nOffset}}),
                     TileInfo({{1, iterInfo.nL0Size}, {0, 0}}));
        tileGraphNodes.scaleTensorPtr =
            LinkScale(function, tensorGraphNodes, TileInfo({{1, iterInfo.nL0Size}, {0, iterInfo.nOffset}}),
                      TileInfo({{1, iterInfo.nL0Size}, {0, 0}}));
        for (iterInfo.mOffset = 0; iterInfo.mOffset < tileInfo.mView; iterInfo.mOffset += tileInfo.tileML0) {
            iterInfo.mL0Size = std::min(tileInfo.mView - iterInfo.mOffset, tileInfo.tileML0);
            tileGraphNodes.outTensorPtr =
                cTensorPtr->View(function, {iterInfo.mL0Size, iterInfo.nL0Size}, {iterInfo.mOffset, iterInfo.nOffset});
            LogicalTensorPtr aL1TensorPtr = nullptr;
            LogicalTensorPtr bL1TensorPtr = nullptr;
            LogicalTensorPtr aScaleL1TensorPtr = nullptr;
            LogicalTensorPtr bScaleL1TensorPtr = nullptr;
            for (iterInfo.kOffset = 0; iterInfo.kOffset < tileInfo.kView; iterInfo.kOffset += tileInfo.tileKL0) {
                UpdateIterInfo(tileInfo, iterInfo);
                tileGraphNodes.aTensorPtr =
                    LinkTensorA(function, tensorGraphNodes, attrParam, tileInfo, iterInfo, aL1TensorPtr);
                tileGraphNodes.bTensorPtr =
                    LinkTensorB(function, tensorGraphNodes, attrParam, tileInfo, iterInfo, bL1TensorPtr);
                if (attrParam.hasMXScale) {
                    tileGraphNodes.aScaleTensorPtr =
                        LinkTensorAScale(function, tensorGraphNodes, attrParam, tileInfo, iterInfo, aScaleL1TensorPtr);
                    tileGraphNodes.bScaleTensorPtr =
                        LinkTensorBScale(function, tensorGraphNodes, attrParam, tileInfo, iterInfo, bScaleL1TensorPtr);
                }
                LinkAMulB(function, tensorGraphNodes, attrParam, iterInfo, tileGraphNodes);
            }
        }
    }
    MATMUL_LOGD("ConstructTileGraph: Finish.");
}

void AddAMulBNode(const MatmulGraphNodes &tensorGraphNodes, const MatmulAttrParam &attrParam,
    const MatmulExtendParam &extendParam = {}) {
    if (CheckValidShape(tensorGraphNodes.aTensorPtr) && CheckValidShape(tensorGraphNodes.bTensorPtr)) {
        SymbolicScalar mSizeDyn = attrParam.transA ? tensorGraphNodes.aTensorPtr->GetDynValidShape()[1] :
                                                     tensorGraphNodes.aTensorPtr->GetDynValidShape()[0];
        SymbolicScalar kSizeDyn = attrParam.transA ? tensorGraphNodes.aTensorPtr->GetDynValidShape()[0] :
                                                     tensorGraphNodes.aTensorPtr->GetDynValidShape()[1];
        SymbolicScalar nSizeDyn = attrParam.transB ? tensorGraphNodes.bTensorPtr->GetDynValidShape()[0] :
                                                     tensorGraphNodes.bTensorPtr->GetDynValidShape()[1];
        ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.outTensorPtr != nullptr)
            << "cTensorPtr is nullptr.";
        tensorGraphNodes.outTensorPtr->UpdateDynValidShape({mSizeDyn, nSizeDyn});
    }

    std::vector<LogicalTensorPtr> operandVec = {tensorGraphNodes.aTensorPtr, tensorGraphNodes.bTensorPtr};
    bool gmAccumulationFlag = false;
    if (attrParam.hasMXScale) {
        operandVec.push_back(tensorGraphNodes.aScaleTensorPtr);
        operandVec.push_back(tensorGraphNodes.bScaleTensorPtr);
    }
    if (tensorGraphNodes.gmAccumulationTensorPtr != nullptr) {
        operandVec.push_back(tensorGraphNodes.gmAccumulationTensorPtr);
        gmAccumulationFlag = true;
    }
    if (extendParam.biasTensor.GetStorage() != nullptr) {
        operandVec.push_back(extendParam.biasTensor.GetStorage());
    }
    if (extendParam.scaleTensor.GetStorage() != nullptr) {
        operandVec.push_back(extendParam.scaleTensor.GetStorage());
    }
    Function *functionPtr = Program::GetInstance().GetCurrentFunction();

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, functionPtr != nullptr) << "functionPtr is nullptr.";
    auto &op = functionPtr->AddOperation(Opcode::OP_A_MUL_B, operandVec, {tensorGraphNodes.outTensorPtr});
    SetTensorGraphAttr(op, extendParam, gmAccumulationFlag, attrParam);
}

Tensor ConstructTensorGraph(DataType dataType, MatmulGraphNodes &tensorGraphNodes, const MatmulAttrParam &attrParam,
    const MatmulExtendParam &param = {}) {
    MATMUL_LOGD("ConstructTensorGraph: Start.");
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.aTensorPtr != nullptr) << "aTensorPtr is nullptr.";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, tensorGraphNodes.bTensorPtr != nullptr) << "bTensorPtr is nullptr.";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.aTensorPtr->GetShape().size() >= SHAPE_DIM2)
        << "The dimension of aTensor must be >= 2! The dimension of aTensor: "
        << tensorGraphNodes.aTensorPtr->GetShape().size();
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, tensorGraphNodes.bTensorPtr->GetShape().size() >= SHAPE_DIM2)
        << "The dimension of bTensor must be >= 2! The dimension of bTensor: "
        << tensorGraphNodes.bTensorPtr->GetShape().size();
    int64_t mSize =
        attrParam.transA ? tensorGraphNodes.aTensorPtr->GetShape()[1] : tensorGraphNodes.aTensorPtr->GetShape()[0];
    int64_t kSizeA =
        attrParam.transA ? tensorGraphNodes.aTensorPtr->GetShape()[0] : tensorGraphNodes.aTensorPtr->GetShape()[1];
    int64_t kSizeB =
        attrParam.transB ? tensorGraphNodes.bTensorPtr->GetShape()[1] : tensorGraphNodes.bTensorPtr->GetShape()[0];
    int64_t nSize =
        attrParam.transB ? tensorGraphNodes.bTensorPtr->GetShape()[0] : tensorGraphNodes.bTensorPtr->GetShape()[1];

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kSizeA == kSizeB)
        << "Matrix K dimension mismatch, kSizeA: " << kSizeA << ", kSizeB: " << kSizeB;
    Tensor cMatrix(dataType, {mSize, nSize}, "TensorC");
    if (attrParam.isCMatrixNZ) {
        ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, BytesOf(dataType) > 0)
            << "BytesOf(dataType): " << BytesOf(dataType) << ". Must be positive.";
        int64_t c0Size = dataType == DataType::DT_INT32 ? ALIGN_SIZE_16 : ALIGN_SIZE_32 / BytesOf(dataType);
        cMatrix = Tensor(dataType, {mSize, CeilAlign(nSize, c0Size)}, "TensorC", TileOpFormat::TILEOP_NZ);
    }
    tensorGraphNodes.outTensorPtr = cMatrix.GetStorage();
    AddAMulBNode(tensorGraphNodes, attrParam, param);
    return cMatrix;
}

// 根据UB大小设置VecTile
static void SetVecTileBasedOnUbSize(DataType outType, const CubeTile &cubeTile) {
    uint64_t ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    // Add的两个输入矩阵总大小不能超过UB大小限制
    if (cubeTile.m[0] * cubeTile.n[0] * BytesOf(outType) * 2 <= ubSize || outType == DT_INT32) {
        TileShape::Current().SetVecTile({cubeTile.m[0], cubeTile.n[0]});
    } else {
        TileShape::Current().SetVecTile({128, 128});
    }
}

static Tensor AssembleGmAccumulationTensor(DataType outType, const Tensor gmAccumulationTensor,
    std::vector<int64_t> outSize, std::vector<SymbolicScalar> validShape, bool isCMatrixNZ) {
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, outSize.size() == SHAPE_DIM2 && validShape.size() == SHAPE_DIM2)
        << "Both outSize and validShape must be 2-element vectors";

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, outSize[0] != 0 && outSize[1] != 0) << "Matrix size cannot be 0";
    Tensor assembleTensor(
        outType, {outSize[0], outSize[1]}, "", isCMatrixNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, assembleTensor.GetStorage() != nullptr)
        << "Can not get assembleTensor's storage";

    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, gmAccumulationTensor.GetStorage() != nullptr)
        << "Can not get gmAccumulationTensor's storage";
    gmAccumulationTensor.GetStorage()->UpdateDynValidShape({validShape[0], validShape[1]});
    assembleTensor.GetStorage()->UpdateDynValidShape({validShape[0], validShape[1]});
    Assemble(gmAccumulationTensor, {0, 0}, assembleTensor);
    return assembleTensor;
}

static Tensor GetGmDeterministicAccumulationTensor(std::vector<Tensor> gmPartialSums, int64_t kLoop) {
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, gmPartialSums.size() == static_cast<uint64_t>(kLoop))
        << "GmPartialSums' size mismatch kLoop.";
    for (int64_t kIdx = 1; kIdx < kLoop; ++kIdx) {
        gmPartialSums[0] = npu::tile_fwk::Add(gmPartialSums[0], gmPartialSums[kIdx]);
    }
    return gmPartialSums[0];
}

static Tensor GetGmAtomicAccumulationTensor(DataType outType, Tensor gmAccumulationTensor,
    std::vector<Tensor> gmPartialSums, std::vector<int64_t> outSize, std::vector<SymbolicScalar> validShape,
    bool isCMatrixNZ) {
    gmAccumulationTensor = npu::tile_fwk::Reduce(gmPartialSums, ReduceMode::ATOMIC_ADD);
    return AssembleGmAccumulationTensor(outType, gmAccumulationTensor, outSize, validShape, isCMatrixNZ);
}

static Tensor ConstructGmAccumulationTensorGraph(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix,
    const MatmulAttrParam &attrParam, const MatmulExtendParam &extendParam = {}) {
    auto &cubeTile = TileShape::Current().GetCubeTile();
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, aMatrix.GetStorage() != nullptr && bMatrix.GetStorage() != nullptr)
        << "Both aMatrix and bMatrix cannot get storage";
    auto aMatrixValidShape = aMatrix.GetStorage()->GetDynValidShape();
    auto bMatrixValidShape = bMatrix.GetStorage()->GetDynValidShape();
    SymbolicScalar mValidShape = attrParam.transA ? aMatrixValidShape[1] : aMatrixValidShape[0];
    SymbolicScalar nValidShape = attrParam.transB ? bMatrixValidShape[0] : bMatrixValidShape[1];
    SymbolicScalar kL1TileShape = std::min(cubeTile.k[1], cubeTile.k[2]);
    int64_t mSize = attrParam.transA ? aMatrix.GetShape()[1] : aMatrix.GetShape()[0];
    int64_t kSize = attrParam.transA ? aMatrix.GetShape()[0] : aMatrix.GetShape()[1];
    int64_t nSize = attrParam.transB ? bMatrix.GetShape()[0] : bMatrix.GetShape()[1];

    SetVecTileBasedOnUbSize(outType, cubeTile);
    Tensor gmAccumulationTensor = (outType == DT_INT32) ? Full(Element(outType, static_cast<int64_t>(0)), outType,
                                                              {mSize, nSize}, {mValidShape, nValidShape}) :
                                                          Tensor();
    std::vector<Tensor> gmPartialSums;
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, kL1TileShape != 0) << "kL1TileShape can not be 0";
    const int64_t kLoop = (kSize + kL1TileShape - 1) / kL1TileShape;
    const int64_t kL1Size = std::min(kSize, kL1TileShape);
    for (int64_t kIdx = 0; kIdx < kLoop; ++kIdx) {
        int64_t kValidshape = std::min(kSize - kL1Size * kIdx, kL1Size);
        Tensor tensorA;
        if (attrParam.transA) {
            tensorA = View(aMatrix, {kL1Size, mSize}, {kValidshape, mValidShape}, {kL1Size * kIdx, 0});
        } else {
            tensorA = View(aMatrix, {mSize, kL1Size}, {mValidShape, kValidshape}, {0, kL1Size * kIdx});
        }
        Tensor tensorB;
        if (attrParam.transB) {
            tensorB = View(bMatrix, {nSize, kL1Size}, {nValidShape, kValidshape}, {0, kL1Size * kIdx});
        } else {
            tensorB = View(bMatrix, {kL1Size, nSize}, {kValidshape, nValidShape}, {kL1Size * kIdx, 0});
        }
        MatmulGraphNodes tensorGraphNodes(
            tensorA.GetStorage(), tensorB.GetStorage(), gmAccumulationTensor.GetStorage());
        Tensor gmPartialSum = ConstructTensorGraph(outType, tensorGraphNodes, attrParam, extendParam);
        gmPartialSums.emplace_back(gmPartialSum);
    }
    if (outType == DT_INT32) {
        return GetGmAtomicAccumulationTensor(outType, gmAccumulationTensor, gmPartialSums, {mSize, nSize},
            {mValidShape, nValidShape}, attrParam.isCMatrixNZ);
    } else {
        return GetGmDeterministicAccumulationTensor(gmPartialSums, kLoop);
    }
}

Tensor Matmul(
    DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans, bool isBTrans, bool isCMatrixNZ) {
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    MATMUL_LOGD("Matmul[Basic]: Start.");
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "Matmul operands check failed";
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), bMatrix.GetStorage());
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul: Using GM accumulation mode.");
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
}

Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const MatmulExtendParam &param,
    bool isATrans, bool isBTrans, bool isCMatrixNZ) {
    MATMUL_LOGD("Matmul[Extend]: Start.");
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam, param);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "Matmul operands check failed";
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), bMatrix.GetStorage());
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul: Using GM accumulation mode.");
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam, param);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam, param);
}

Tensor MatmulMX(DataType outType, const Tensor &aMatrix, const Tensor &aScale, const Tensor &bMatrix,
    const Tensor &bScale, bool isATrans, bool isAScaleTrans, bool isBTrans, bool isBScaleTrans, bool isCMatrixNZ) {
    MATMUL_LOGD("MatmulMX[Basic]: Start.");
    MatmulAttrParam attrParam(isATrans, isAScaleTrans, isBTrans, isBScaleTrans, isCMatrixNZ);
    CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "Matmul operands check failed";
    Status checkMXStatus = CheckMXMatmulOperands(aMatrix, aScale, bMatrix, bScale, attrParam);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkMXStatus == SUCCESS) << "MXMatmul operands check failed";
    MatmulGraphNodes tensorGraphNodes(
        aMatrix.GetStorage(), aScale.GetStorage(), bMatrix.GetStorage(), bScale.GetStorage());
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
}

Tensor MatmulMX(DataType outType, const Tensor &aMatrix, const Tensor &aScale, const Tensor &bMatrix,
    const Tensor &bScale, const MatmulExtendParam &param, bool isATrans, bool isAScaleTrans, bool isBTrans,
    bool isBScaleTrans, bool isCMatrixNZ) {
    MATMUL_LOGD("MatmulMX[Extend]: Start.");
    MatmulAttrParam attrParam(isATrans, isAScaleTrans, isBTrans, isBScaleTrans, isCMatrixNZ);
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam, param);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkStatus == SUCCESS) << "Matmul operands check failed";
    Status checkMXStatus = CheckMXMatmulOperands(aMatrix, aScale, bMatrix, bScale, attrParam);
    ASSERT(MatmulErrorCode::ERR_RUNTIME_LOGIC, checkMXStatus == SUCCESS) << "MXMatmul operands check failed";
    MatmulGraphNodes tensorGraphNodes(
        aMatrix.GetStorage(), aScale.GetStorage(), bMatrix.GetStorage(), bScale.GetStorage());
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam, param);
}

void CheckABatchMulB(const Tensor &operand1, const Tensor &operand2) {
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, operand1.GetStorage() != nullptr) << "A Tensor cannot be null";
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, operand2.GetStorage() != nullptr) << "B Tensor cannot be null";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        operand1.GetShape().size() == SHAPE_DIM3 || operand1.GetShape().size() == SHAPE_DIM4)
        << "Batch matmul only support 3 dimensions or 4 dimensions.";

    for (uint64_t bIdx = 0; bIdx < operand1.GetShape().size() - SHAPE_DIM2; bIdx++) {
        const int64_t batchSizeA = operand1.GetShape()[bIdx];
        const int64_t batchSizeB = operand2.GetShape()[bIdx];
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, batchSizeA == batchSizeB || batchSizeB == 1 || batchSizeA == 1)
            << "batchSize invalid: A" << bIdx << "= B" << bIdx << "or 1 allowed. A" << bIdx << ": " << batchSizeA
            << ", B" << bIdx << ": " << batchSizeB;
    }
}

Tensor ConstructBatchMatmulTensorGraph3D(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    const int64_t batchSizeA = operand1.GetShape()[0];
    const int64_t batchSizeB = operand2.GetShape()[0];
    const int64_t batchSize = std::max(batchSizeA, batchSizeB);

    const int64_t mView = attrParam.transA ? operand1.GetShape()[SHAPE_DIM2] : operand1.GetShape()[1];
    const int64_t nView = attrParam.transB ? operand2.GetShape()[1] : operand2.GetShape()[SHAPE_DIM2];
    Tensor result = attrParam.isCMatrixNZ ?
                        Tensor(dataType, {batchSize, mView, nView}, "BatchMatmulOutputNz", TileOpFormat::TILEOP_NZ) :
                        Tensor(dataType, {batchSize, mView, nView});
    auto oriVecTile = TileShape::Current().GetVecTile();
    TileShape::Current().SetVecTile({1, 128, 128});
    for (int64_t bIdx = 0; bIdx < batchSize; bIdx++) {
        int64_t offsetBatchA = batchSizeA == 1 ? 0 : bIdx;
        int64_t offsetBatchB = batchSizeB == 1 ? 0 : bIdx;
        auto aValidShape3D = operand1.GetStorage()->GetDynValidShape();
        auto bValidShape3D = operand2.GetStorage()->GetDynValidShape();

        Tensor aTensorSingleBatch = View(operand1, {1, operand1.GetShape()[1], operand1.GetShape()[SHAPE_DIM2]},
            std::vector<SymbolicScalar>({1, aValidShape3D[1], aValidShape3D[SHAPE_DIM2]}), {offsetBatchA, 0, 0});
        Tensor bTensorSingleBatch = View(operand2, {1, operand2.GetShape()[1], operand2.GetShape()[SHAPE_DIM2]},
            std::vector<SymbolicScalar>({1, bValidShape3D[1], bValidShape3D[SHAPE_DIM2]}), {offsetBatchB, 0, 0});

        Tensor aTensor = Reshape(aTensorSingleBatch, {operand1.GetShape()[1], operand1.GetShape()[SHAPE_DIM2]},
            std::vector<SymbolicScalar>({aValidShape3D[1], aValidShape3D[SHAPE_DIM2]}));
        Tensor bTensor = Reshape(bTensorSingleBatch, {operand2.GetShape()[1], operand2.GetShape()[SHAPE_DIM2]},
            std::vector<SymbolicScalar>({bValidShape3D[1], bValidShape3D[SHAPE_DIM2]}));
        Tensor cTensor(dataType, {mView, nView}, "cTensorSingleBatch");

        MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), bTensor.GetStorage());
        tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
        AddAMulBNode(tensorGraphNodes, attrParam);
        auto cValidShape2D = cTensor.GetStorage()->GetDynValidShape();
        Tensor cTensor3D = Reshape(cTensor, {1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
            std::vector<SymbolicScalar>({1, cValidShape2D[0], cValidShape2D[1]}));
        Assemble(cTensor3D, {bIdx, 0, 0}, result);
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

Tensor ConstructBatchMatmulTensorGraph4D(
    DataType dataType, const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    const int64_t batchSizeA1 = operand1.GetShape()[0];
    const int64_t batchSizeA2 = operand1.GetShape()[1];
    const int64_t batchSizeB1 = operand2.GetShape()[0];
    const int64_t batchSizeB2 = operand2.GetShape()[1];
    const int64_t batchSize1 = std::max(batchSizeA1, batchSizeB1);
    const int64_t batchSize2 = std::max(batchSizeA2, batchSizeB2);
    const int64_t mView = attrParam.transA ? operand1.GetShape()[SHAPE_DIM3] : operand1.GetShape()[SHAPE_DIM2];
    const int64_t nView = attrParam.transB ? operand2.GetShape()[SHAPE_DIM2] : operand2.GetShape()[SHAPE_DIM3];
    Tensor result = attrParam.isCMatrixNZ ? Tensor(dataType, {batchSize1, batchSize2, mView, nView}, "BatchMatmulOutputNz",
                                      TileOpFormat::TILEOP_NZ) :
                                  Tensor(dataType, {batchSize1, batchSize2, mView, nView});
    auto oriVecTile = TileShape::Current().GetVecTile();
    TileShape::Current().SetVecTile({1, 1, 128, 128});
    for (int64_t bIdx1 = 0; bIdx1 < batchSize1; bIdx1++) {
        int64_t offsetBatchA1 = batchSizeA1 == 1 ? 0 : bIdx1;
        int64_t offsetBatchB1 = batchSizeB1 == 1 ? 0 : bIdx1;
        for (int64_t bIdx2 = 0; bIdx2 < batchSize2; bIdx2++) {
            int64_t offsetBatchA2 = batchSizeA2 == 1 ? 0 : bIdx2;
            int64_t offsetBatchB2 = batchSizeB2 == 1 ? 0 : bIdx2;
            auto aValidShape4D = operand1.GetStorage()->GetDynValidShape();
            auto bValidShape4D = operand2.GetStorage()->GetDynValidShape();

            Tensor aTensorSingleBatch =
                View(operand1, {1, 1, operand1.GetShape()[SHAPE_DIM2], operand1.GetShape()[SHAPE_DIM3]},
                    std::vector<SymbolicScalar>({1, 1, aValidShape4D[SHAPE_DIM2], aValidShape4D[SHAPE_DIM3]}),
                    {offsetBatchA1, offsetBatchA2, 0, 0});
            Tensor bTensorSingleBatch =
                View(operand2, {1, 1, operand2.GetShape()[SHAPE_DIM2], operand2.GetShape()[SHAPE_DIM3]},
                    std::vector<SymbolicScalar>({1, 1, bValidShape4D[SHAPE_DIM2], bValidShape4D[SHAPE_DIM3]}),
                    {offsetBatchB1, offsetBatchB2, 0, 0});
            Tensor aTensor =
                Reshape(aTensorSingleBatch, {operand1.GetShape()[SHAPE_DIM2], operand1.GetShape()[SHAPE_DIM3]},
                    std::vector<SymbolicScalar>({aValidShape4D[SHAPE_DIM2], aValidShape4D[SHAPE_DIM3]}));
            Tensor bTensor =
                Reshape(bTensorSingleBatch, {operand2.GetShape()[SHAPE_DIM2], operand2.GetShape()[SHAPE_DIM3]},
                    std::vector<SymbolicScalar>({bValidShape4D[SHAPE_DIM2], bValidShape4D[SHAPE_DIM3]}));
            Tensor cTensor(dataType, {mView, nView}, "cTensorSingleBatch");

            MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), bTensor.GetStorage());
            tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
            AddAMulBNode(tensorGraphNodes, attrParam);
            auto cValidShape2D = cTensor.GetStorage()->GetDynValidShape();
            Tensor cTensor4D = Reshape(cTensor, {1, 1, cTensor.GetShape()[0], cTensor.GetShape()[1]},
                std::vector<SymbolicScalar>({1, 1, cValidShape2D[0], cValidShape2D[1]}));
            Assemble(cTensor4D, {bIdx1, bIdx2, 0, 0}, result);
        }
    }
    TileShape::Current().SetVecTile(oriVecTile);
    return result;
}

Tensor BatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix, const bool isTransA,
    const bool isTransB, const bool isCMatrixNZ) {
    MatmulAttrParam attrParam(isTransA, isTransB, isCMatrixNZ);
    CheckMatmulOperands(dataType, aMatrix, bMatrix, attrParam);
    CheckABatchMulB(aMatrix, bMatrix);
    if (aMatrix.GetShape().size() == SHAPE_DIM4) {
        return ConstructBatchMatmulTensorGraph4D(dataType, aMatrix, bMatrix, attrParam);
    } else {
        return ConstructBatchMatmulTensorGraph3D(dataType, aMatrix, bMatrix, attrParam);
    }
}

// 定制接口：用于Transpose + BMM + Transpose融合场景
// 当前仅支持：(M, B, K) @ (B, K, N) -> (M, B, N)
Tensor TransposedBatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix) {
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID,
        aMatrix.GetShape().size() == SHAPE_DIM3 && bMatrix.GetShape().size() == SHAPE_DIM3)
        << "TransposedBatchMatmul only support 3-dim inputs, aMatrix dim: " << aMatrix.GetShape().size()
        << ", bMatrix dim: " << bMatrix.GetShape().size();
    const int64_t mSize = aMatrix.GetShape()[0];
    const int64_t batchSizeA = aMatrix.GetShape()[1];
    const int64_t kaSize = aMatrix.GetShape()[SHAPE_DIM2];
    const int64_t batchSizeB = bMatrix.GetShape()[0];
    const int64_t kbSize = bMatrix.GetShape()[1];
    const int64_t nSize = bMatrix.GetShape()[SHAPE_DIM2];
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, batchSizeA == batchSizeB)
        << "batchSize invalid, expect batchSizeA = batchSizeB, given batchSizeA: " << batchSizeA
        << ", batchSizeB: " << batchSizeB;

    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, kaSize == kbSize)
        << "kSize invalid, expect kaSize = kbSize, given kaSize: " << kaSize << ", kbSize: " << kbSize;

    // 128: custom tile shape size
    TileShape::Current().SetVecTile({1, 128, 128});
    Tensor aMatrixFused = Reshape(aMatrix, {mSize, batchSizeA * kaSize});
    Tensor cMatrix(dataType, {mSize, batchSizeA * nSize});
    for (int64_t bIdx = 0; bIdx < batchSizeA; ++bIdx) {
        Tensor aTensor =
            View(aMatrixFused, {mSize, kaSize}, std::vector<SymbolicScalar>({mSize, kaSize}), {0, bIdx * kaSize});
        Tensor bTensorSingleBatch =
            View(bMatrix, {1, kbSize, nSize}, std::vector<SymbolicScalar>({1, kbSize, nSize}), {bIdx, 0, 0});
        Tensor bTensor = Reshape(bTensorSingleBatch, {kbSize, nSize});
        Tensor cTensor(dataType, {mSize, nSize}, "TensorC");
        MatmulAttrParam attrParam(false, false, false);
        MatmulGraphNodes tensorGraphNodes(aTensor.GetStorage(), bTensor.GetStorage());
        tensorGraphNodes.outTensorPtr = cTensor.GetStorage();
        AddAMulBNode(tensorGraphNodes, attrParam);
        Assemble(cTensor, {0, bIdx * nSize}, cMatrix);
    }
    return Reshape(cMatrix, {mSize, batchSizeA, nSize});
}
}  // namespace Matrix
}  // namespace tile_fwk
}  // namespace npu
