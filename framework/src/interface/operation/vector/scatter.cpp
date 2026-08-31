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
 * \\file scatter.cpp
 * \\brief
 */

#include <climits>
#include <limits>
#include <cmath>
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

struct ScatterElementSPara {
    const LogicalTensorPtr& dstTensor;
    const LogicalTensorPtr& srcInput;
    const LogicalTensorPtr& idxInput;
    const Element& scalar;
    const int axis;
    const int scatterMode;
};

struct ScatterElementSTileInfoPara {
    TileInfo srcTileInfo;
    TileInfo idxTileInfo;
    TileInfo dstTileInfo;
};

void InnerTiledScatterElementS(size_t cur, Function& function, const TileShape& tileShape,
                               const ScatterElementSPara& scatterPara, ScatterElementSTileInfoPara& scatterTileInfo)
{
    const LogicalTensorPtr& dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr& srcInput = scatterPara.srcInput;
    const LogicalTensorPtr& idxInput = scatterPara.idxInput;
    const Element& scalar = scatterPara.scalar;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto srcTile = srcInput->View(function, scatterTileInfo.srcTileInfo.shape, scatterTileInfo.srcTileInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxTileInfo.shape, scatterTileInfo.idxTileInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstTileInfo.shape, scatterTileInfo.dstTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_SCATTER_ELEMENT, {srcTile, idxTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= dstTensor->shape[axis])
        << "The axis is not supported for tile splitting";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= idxInput->shape[axis])
        << "The axis is not supported for tile splitting";
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = std::max(dstTensor->shape[axis], idxInput->shape[axis]);
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxTileInfo.offset[cur] = 0;
            scatterTileInfo.idxTileInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstTileInfo.offset[cur] = 0;
            scatterTileInfo.dstTileInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcTileInfo.offset[cur] = 0;
            scatterTileInfo.srcTileInfo.shape[cur] = srcInput->shape[cur];
        } else {
            scatterTileInfo.idxTileInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.dstTileInfo.offset[cur] = i;
            scatterTileInfo.dstTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.srcTileInfo.offset[cur] = i;
            scatterTileInfo.srcTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
        }
        InnerTiledScatterElementS(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatterElementS(Function& function, const TileShape& tileShape, const ScatterElementSPara& scatterPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The size of srcInput shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The size of idxInput shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The size of dst shape and offset should be equal";

    ScatterElementSTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
    };
    InnerTiledScatterElementS(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatterElementS(Function& function, const ScatterElementSPara& scatterPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER_ELEMENT,
                                           {scatterPara.srcInput, scatterPara.idxInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, scatterPara.scalar);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

static void CheckScatterElementSParamsInvalid(const Tensor& self, const Tensor& indices, int axis,
                                              const ScatterMode reduce)
{
    static const std::unordered_set<DataType> SCATTER_A2A3_TYPES = {DT_FP32,  DT_FP16,  DT_BF16,  DT_INT8,
                                                                    DT_UINT8, DT_INT16, DT_INT32, DT_INT64};
    static const std::unordered_set<DataType> SCATTER_A5_TYPES = {DT_FP32,  DT_FP16,  DT_BF16,  DT_INT8,
                                                                  DT_UINT8, DT_INT16, DT_INT32, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(SCATTER_A2A3_TYPES, SCATTER_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SCATTER");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "SCATTER");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), indices.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), indices.GetStorage(), "SCATTER");
    CheckAxisRange(self, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, reduce <= ScatterMode::UNKNOWN)
        << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= self.GetShape()[i])
            << "The shape of indices and self should be equal";
    }
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SCATTER");
    CheckTensorShapeSize(self.GetStorage(), "SCATTER");
    CheckTensorShapeSize(indices.GetStorage(), "SCATTER");
}

Tensor Scatter(const Tensor& self, const Tensor& indices, const Element& src, int axis, ScatterMode reduce)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");

    CheckScatterElementSParamsInvalid(self, indices, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    DataType orgDtype = self.GetDataType();
    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }
    axis = axis < 0 ? operandCast.GetShape().size() + axis : axis;
    Tensor result(operandCast.GetStorage()->Datatype(), operandCast.GetShape());
    result.GetStorage()->UpdateDynValidShape(operandCast.GetStorage()->GetDynValidShape());
    CALL(ScatterElementS, *Program::GetInstance().GetCurrentFunction(),
         {result.GetStorage(), operandCast.GetStorage(), indices.GetStorage(), src, axis, static_cast<int>(reduce)});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

struct ScatterPara {
    const LogicalTensorPtr& dstTensor;
    const LogicalTensorPtr& selfInput;
    const LogicalTensorPtr& idxInput;
    const LogicalTensorPtr& srcInput;
    const int axis;
    const int scatterMode;
};

struct ScatterTileInfoPara {
    TileInfo srcInfo;
    TileInfo idxInfo;
    TileInfo dstInfo;
    TileInfo selfInfo;
};

void InnerTiledScatter(size_t cur, Function& function, const TileShape& tileShape, const ScatterPara& scatterPara,
                       ScatterTileInfoPara& scatterTileInfo)
{
    const LogicalTensorPtr& dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr& selfInput = scatterPara.selfInput;
    const LogicalTensorPtr& idxInput = scatterPara.idxInput;
    const LogicalTensorPtr& srcInput = scatterPara.srcInput;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto selfTile = selfInput->View(function, scatterTileInfo.selfInfo.shape, scatterTileInfo.selfInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxInfo.shape, scatterTileInfo.idxInfo.offset);
        auto srcTile = srcInput->View(function, scatterTileInfo.srcInfo.shape, scatterTileInfo.srcInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstInfo.shape, scatterTileInfo.dstInfo.offset);
        Shape tmpShape({idxTile->GetShape()[idxTile->GetShape().size() - 1]});
        auto tmpBuffer = std::make_shared<LogicalTensor>(function, idxTile->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_SCATTER, {selfTile, idxTile, srcTile}, {dstTile, tmpBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= dstTensor->shape[axis])
        << "The axis is not supported for tile splitting";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= idxInput->shape[axis])
        << "The axis is not supported for tile splitting";
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = std::max(dstTensor->shape[axis], idxInput->shape[axis]);
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxInfo.offset[cur] = 0;
            scatterTileInfo.idxInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstInfo.offset[cur] = 0;
            scatterTileInfo.dstInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcInfo.offset[cur] = 0;
            scatterTileInfo.srcInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.selfInfo.offset[cur] = 0;
            scatterTileInfo.selfInfo.shape[cur] = selfInput->shape[cur];
        } else {
            scatterTileInfo.idxInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.dstInfo.offset[cur] = i;
            scatterTileInfo.dstInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.srcInfo.offset[cur] = i;
            scatterTileInfo.srcInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.selfInfo.offset[cur] = i;
            scatterTileInfo.selfInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                           tmpTile);
        }
        InnerTiledScatter(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatter(Function& function, const TileShape& tileShape, const ScatterPara& scatterPara)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The shape size of srcInput and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The shape size of idxInput and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The shape size of dst and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.selfInput->shape.size() == scatterPara.selfInput->offset.size())
        << "The shape size of selfInput and offset should be equal";

    ScatterTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
        TileInfo(scatterPara.selfInput->shape.size(), scatterPara.selfInput->offset.size()),
    };
    InnerTiledScatter(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatter(Function& function, const ScatterPara& scatterPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER,
                                           {scatterPara.selfInput, scatterPara.idxInput, scatterPara.srcInput},
                                           {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
}

static void CheckScatterParamsInvalid(const Tensor& self, const Tensor& indices, const Tensor& src, int axis,
                                      const ScatterMode reduce)
{
    static const std::unordered_set<DataType> SCATTER_A2A3_TYPES = {DT_FP32,  DT_FP16,  DT_BF16,  DT_INT8,
                                                                    DT_UINT8, DT_INT16, DT_INT32, DT_INT64};
    static const std::unordered_set<DataType> SCATTER_A5_TYPES = {DT_FP32,  DT_FP16,  DT_BF16,  DT_INT8,
                                                                  DT_UINT8, DT_INT16, DT_INT32, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(SCATTER_A2A3_TYPES, SCATTER_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SCATTER");
    CheckTensorsDataTypeConsistency(self.GetStorage(), src.GetStorage(), "SCATTER");
    std::unordered_set<DataType> indexSupportedTypes = {DT_INT32, DT_INT64};
    CheckTensorDataType(indices.GetStorage(), indexSupportedTypes, "SCATTER");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), indices.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), indices.GetStorage(), "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), src.GetStorage(), "SCATTER");
    CheckTensorsFormatConsistency(indices.GetStorage(), src.GetStorage(), "SCATTER");
    CheckAxisRange(self, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, reduce <= ScatterMode::UNKNOWN)
        << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= src.GetShape()[i])
            << "The shape size of src and indices should be equal";
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= self.GetShape()[i])
            << "The shape size of src and indices should be equal";
    }
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SCATTER");
    CheckTensorShapeSize(self.GetStorage(), "SCATTER");
    CheckTensorShapeSize(indices.GetStorage(), "SCATTER");
    CheckTensorShapeSize(src.GetStorage(), "SCATTER");
}

Tensor Scatter(const Tensor& self, const Tensor& indices, const Tensor& src, int axis, ScatterMode reduce)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");
    CheckTensorFormat(indices.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Scatter");

    CheckScatterParamsInvalid(self, indices, src, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    DataType orgDtype = self.GetDataType();
    auto operandSelfCast = Tensor(DataType::DT_FP32, self.GetShape());
    auto operandSrcCast = Tensor(DataType::DT_FP32, src.GetShape());
    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        operandSelfCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                               self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
        operandSrcCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                              src.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandSelfCast = self;
        operandSrcCast = src;
    }
    axis = axis < 0 ? operandSelfCast.GetShape().size() + axis : axis;
    Tensor result(operandSelfCast.GetStorage()->Datatype(), operandSelfCast.GetShape());
    result.GetStorage()->UpdateDynValidShape(operandSelfCast.GetStorage()->GetDynValidShape());
    CALL(Scatter, *Program::GetInstance().GetCurrentFunction(),
         {result.GetStorage(), operandSelfCast.GetStorage(), indices.GetStorage(), operandSrcCast.GetStorage(), axis,
          static_cast<int>(reduce)});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

void ScatterElementSOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatterElementS(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], scalar, axis, scatterMode});
}

void ScatterOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatter(function, tileShape,
                 {oOperand[0], iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], axis, scatterMode});
}

REGISTER_OPERATION_TILED_FUNC(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ScatterElementSOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER, Opcode::OP_SCATTER, ScatterOperationTileFunc);

} // namespace npu::tile_fwk
