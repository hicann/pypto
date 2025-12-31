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
 * \file indexing.cpp
 * \brief
 */

#include <climits>
#include <limits>
#include <cmath>
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "tensor_transformation.h"

namespace npu::tile_fwk {


constexpr float FP16_MAX = 65504.0f;

struct IndexAddPara {
    const LogicalTensorPtr &selfInput;
    const LogicalTensorPtr &srcInput;
    const LogicalTensorPtr &indicesInput;
    const LogicalTensorPtr &dstTensor;
    const int axis;
    const Element &alpha;
};

struct IndexAddTileInfoPara {
    TileInfo selfTileInfo;
    TileInfo srcTileInfo;
    TileInfo indicesTileInfo;
    TileInfo dstTileInfo;
};

void IndexAddExpandFunc(Function &function, const IndexAddPara indexaddPara, IndexAddTileInfoPara &indexaddTileInfo) {
    const LogicalTensorPtr &selfInput = indexaddPara.selfInput;
    const LogicalTensorPtr &srcInput = indexaddPara.srcInput;
    const LogicalTensorPtr &indicesInput = indexaddPara.indicesInput;
    const LogicalTensorPtr &dstTensor = indexaddPara.dstTensor;
    const int axis = indexaddPara.axis;
    const Element &alpha = indexaddPara.alpha;

    auto dstTile = dstTensor->View(function, indexaddTileInfo.dstTileInfo.shape, indexaddTileInfo.dstTileInfo.offset);
    auto selfTile =selfInput->View(function, indexaddTileInfo.selfTileInfo.shape, indexaddTileInfo.selfTileInfo.offset);
    auto srcTile = srcInput->View(function, indexaddTileInfo.srcTileInfo.shape, indexaddTileInfo.srcTileInfo.offset);
    indexaddTileInfo.indicesTileInfo.offset = {indexaddTileInfo.srcTileInfo.offset[axis]}; // 按照srcShape所在的axis轴切分
    indexaddTileInfo.indicesTileInfo.shape = {indexaddTileInfo.srcTileInfo.shape[axis]};
    auto indexTile =indicesInput->View(function, indexaddTileInfo.indicesTileInfo.shape, indexaddTileInfo.indicesTileInfo.offset);

    if (selfTile->Datatype() == DT_INT8) { // vector指令不支持int8的直接计算
        LogicalTensorPtr selfConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP16, selfTile->GetShape());
        Operation &castSelfOp = function.AddOperation(Opcode::OP_CAST, {selfTile}, {selfConvertedTile});
        castSelfOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr srcConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP16, srcTile->GetShape());
        Operation &castSrcOp = function.AddOperation(Opcode::OP_CAST, {srcTile}, {srcConvertedTile});
        castSrcOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr dstConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP16, dstTile->GetShape());

        auto &op = function.AddOperation(Opcode::OP_INDEX_ADD, {selfConvertedTile, srcConvertedTile, indexTile}, {dstConvertedTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        Operation &castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConvertedTile}, {dstTile});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
    } else if (selfTile->Datatype() == DT_BF16) { // vector和scalar均不支持BF16直接计算
        LogicalTensorPtr selfConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, selfTile->GetShape());
        Operation &castSelfOp = function.AddOperation(Opcode::OP_CAST, {selfTile}, {selfConvertedTile});
        castSelfOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr srcConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, srcTile->GetShape());
        Operation &castSrcOp = function.AddOperation(Opcode::OP_CAST, {srcTile}, {srcConvertedTile});
        castSrcOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        LogicalTensorPtr dstConvertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, dstTile->GetShape());

        auto &op = function.AddOperation(Opcode::OP_INDEX_ADD, {selfConvertedTile, srcConvertedTile, indexTile}, {dstConvertedTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        Operation &castDstOp = function.AddOperation(Opcode::OP_CAST, {dstConvertedTile}, {dstTile});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_RINT);
    } else {
        auto &op = function.AddOperation(Opcode::OP_INDEX_ADD, {selfTile, srcTile, indexTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, alpha);
    }
}

void InnerTiledIndexAdd(size_t cur, Function &function, const TileShape &tileShape, const IndexAddPara indexaddPara,
    IndexAddTileInfoPara &indexaddTileInfo) {
    if (cur == indexaddPara.dstTensor->shape.size()) {
        IndexAddExpandFunc(function, indexaddPara, indexaddTileInfo);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    // axis所在轴按照dstShape[axis]进行切分
    if (static_cast<int>(cur) == indexaddPara.axis) {
        tmpTile = indexaddPara.dstTensor->GetShape()[cur];
    }
    // srcInput在axis的维度=indicesInput的维度，且可能比selfInput.shape[axis]大
    for (int i = 0; i < indexaddPara.srcInput->GetShape()[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == indexaddPara.axis) {
            // self和dst不切
            indexaddTileInfo.dstTileInfo.offset[cur] = 0;
            indexaddTileInfo.dstTileInfo.shape[cur] = indexaddPara.dstTensor->shape[cur];
            indexaddTileInfo.selfTileInfo.offset[cur] = 0;
            indexaddTileInfo.selfTileInfo.shape[cur] = indexaddPara.selfInput->shape[cur];
        } else {
            indexaddTileInfo.dstTileInfo.offset[cur] = i;
            indexaddTileInfo.dstTileInfo.shape[cur] = std::min(indexaddPara.dstTensor->shape[cur] - i, tmpTile);
            indexaddTileInfo.selfTileInfo.offset[cur] = i;
            indexaddTileInfo.selfTileInfo.shape[cur] = std::min(indexaddPara.selfInput->shape[cur] - i, tmpTile);
        }
        indexaddTileInfo.srcTileInfo.offset[cur] = i;
        indexaddTileInfo.srcTileInfo.shape[cur] = std::min(indexaddPara.srcInput->GetShape()[cur] - i, tmpTile);
        InnerTiledIndexAdd(cur + 1, function, tileShape, indexaddPara, indexaddTileInfo);
    }
}

void TiledIndexAdd(Function &function, const TileShape &tileShape, const IndexAddPara indexaddPara) {
    // Check Operands Valid
    ASSERT(indexaddPara.selfInput->GetShape().size() == indexaddPara.selfInput->GetOffset().size())
        << "The size of indexaddPara selfinput shape and selfinput offset should be equal";
    ASSERT(indexaddPara.srcInput->GetShape().size() == indexaddPara.srcInput->GetOffset().size())
        << "The size of indexaddPara srcInput shape and srcInput offset should be equal";
    ASSERT(indexaddPara.indicesInput->GetShape().size() == indexaddPara.indicesInput->GetOffset().size())
        << "The size of indexaddPara indicesInput shape and indicesInput offset should be equal";

    IndexAddTileInfoPara indexaddTileInfo{
        TileInfo(indexaddPara.selfInput->GetShape().size(), indexaddPara.selfInput->GetOffset().size()),
        TileInfo(indexaddPara.srcInput->GetShape().size(), indexaddPara.srcInput->GetOffset().size()),
        TileInfo(indexaddPara.indicesInput->GetShape().size(), indexaddPara.indicesInput->GetOffset().size()),
        TileInfo(indexaddPara.dstTensor->GetShape().size(), indexaddPara.dstTensor->GetOffset().size())};
    InnerTiledIndexAdd(0, function, tileShape, indexaddPara, indexaddTileInfo);
}

void TensorIndexAdd(Function &function, const IndexAddPara indexaddPara) {
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_INDEX_ADD,
        {indexaddPara.selfInput, indexaddPara.srcInput, indexaddPara.indicesInput}, {indexaddPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", indexaddPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, indexaddPara.alpha);
}

bool CheckAlphaOverflow(Element alpha, DataType dtype) {
    double value = alpha.Cast<double>();
    if (std::isnan(value) || std::isinf(value))
        return true;
    switch (dtype) {
        case DT_INT8: return value < std::numeric_limits<int8_t>::min() || value > std::numeric_limits<int8_t>::max();
        case DT_INT16:
            return value < std::numeric_limits<int16_t>::min() || value > std::numeric_limits<int16_t>::max();
        case DT_INT32:
            return value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max();
        case DT_FP16: return std::abs(value) > FP16_MAX;
        case DT_BF16: return std::abs(value) > std::numeric_limits<float>::max();
        case DT_FP32: return std::abs(value) > std::numeric_limits<float>::max();
        default: return false;
    }
}

void CheckIndexAddParamsInvalid(
    const Tensor &self, const Tensor &src, const Tensor &indices, const int axis, const Element &alpha) {
    ASSERT(axis < static_cast<int>(self.GetShape().size()) && axis >= -static_cast<int>(self.GetShape().size()))
        << "axis out of range of shape size";
    int axis_ = axis < 0 ? self.GetShape().size() + axis : axis;
    ASSERT(self.GetShape().size() == src.GetShape().size()) << "shape size of self and src should be equal";
    ASSERT(src.GetShape()[axis_] == indices.GetShape()[0]) << "src shape[axis] and indices[0] must equal";
    for (size_t i = 0; i < self.GetShape().size(); ++i) {
        if (static_cast<int>(i) == axis_) {
            continue;
        }
        ASSERT(src.GetShape()[i] == self.GetShape()[i]) << "src shape and self shape should be equal";
    }

    const std::unordered_set<DataType> SRC_SUPPORT_DATATYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8};
    ASSERT(SRC_SUPPORT_DATATYPES.count(self.GetDataType()) > 0) << "The datatype is not supported";
    ASSERT(self.GetDataType() == src.GetDataType()) << "Datatype of src and self should be equal";
    ASSERT(indices.GetDataType() == DT_INT32 || indices.GetDataType() == DT_INT64) << "Datatype of indices is incorrect";
    // 检验 alpha 溢出
    if (CheckAlphaOverflow(alpha, self.GetDataType())) {
        std::string errorMessage =
            "Value cannot be converted to type " + DataType2String(self.GetDataType()) + " without overflow!";
        ASSERT(false) << errorMessage;
    }
}

Tensor IndexAdd(const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
    DECLARE_TRACER();
    Tensor result = self;
    return IndexAdd_(result, src, indices, axis, alpha);
}

Tensor IndexAdd_(const Tensor &self, const Tensor &src, const Tensor &indices, int axis, const Element &alpha) {
    DECLARE_TRACER();
    CheckIndexAddParamsInvalid(self, src, indices, axis, alpha);
    axis = axis < 0 ? self.GetShape().size() + axis : axis;
    DataType selfDataType = self.GetDataType();
    Element alpha_ = Element(selfDataType, alpha.Cast<float>());
    Tensor result(selfDataType, self.GetShape());
    CALL(IndexAdd, *Program::GetInstance().GetCurrentFunction(),
        {self.GetStorage(), src.GetStorage(), indices.GetStorage(), result.GetStorage(), axis, alpha_});

    return result;
}

void TiledGatherOperation(Function &function, const TileShape &tileShape, size_t cur, Input &paramsInput,
    Input &indicesInput, int axis, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile =
            paramsInput.tensor.GetStorage()->View(function, paramsInput.tileInfo.shape, paramsInput.tileInfo.offset);
        auto indicesTile =
            indicesInput.tensor.GetStorage()->View(function, indicesInput.tileInfo.shape, indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (function.IsStatic()) {
            auto &op = function.AddOperation(Opcode::OP_GATHER_FROM_UB, {paramsTile, indicesTile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        } else {
            auto &op = function.AddOperation(Opcode::OP_GATHER, {paramsTile, indicesTile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        }

        return;
    }

    // 按照resultShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur < static_cast<size_t>(axis)) {
            // 在result中gather轴的外层轴
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor.GetShape()[cur];
            paramsInput.tileInfo.shape[cur] =
                std::min(paramsInput.tensor.GetShape()[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
        } else if (cur >= static_cast<size_t>(axis) &&
                   (cur < static_cast<size_t>(axis) + indicesInput.tensor.GetShape().size())) {
            // 当前属于indices的gather轴
            // params[axis]不切
            paramsInput.tileInfo.offset[axis] = 0;
            paramsInput.tileInfo.shape[axis] = paramsInput.tensor.GetShape()[axis];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur - axis] = i % indicesInput.tensor.GetShape()[cur - axis];
            indicesInput.tileInfo.shape[cur - axis] = std::min(
                indicesInput.tensor.GetShape()[cur - axis] - indicesInput.tileInfo.offset[cur - axis], tmpTile);
        } else {
            // 在result中gather轴的内层轴
            int paramHighAxis = cur - indicesInput.tensor.GetShape().size() + 1;
            paramsInput.tileInfo.offset[paramHighAxis] = i % paramsInput.tensor.GetShape()[paramHighAxis];
            paramsInput.tileInfo.shape[paramHighAxis] = std::min(
                paramsInput.tensor.GetShape()[paramHighAxis] - paramsInput.tileInfo.offset[paramHighAxis], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherOperation(function, tileShape, cur + 1, paramsInput, indicesInput, axis, result, resultTileInfo);
    }
}

std::vector<int64_t> GatherOperationResultShape(LogicalTensorPtr params, LogicalTensorPtr indices, int axis) {
    ASSERT(params->shape.size() == params->offset.size()) << "The size of params shape and offset should be equal";
    ASSERT(indices->shape.size() == indices->offset.size()) << "The size of indices shape and offset should be equal";
    int paramsRank = params->shape.size();
    if (axis < 0) {
        axis = axis + paramsRank;
    }
    // result shape: params.shape[:aixs] + indices.shape + params.shape[axis+1:]
    std::vector<int64_t> resultShape = params->shape;
    resultShape.erase(resultShape.begin() + axis);
    resultShape.insert(resultShape.begin() + axis, indices->shape.begin(), indices->shape.end());

    return resultShape;
}

void TiledGatherOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &params,
    const LogicalTensorPtr &indices, int axis, const LogicalTensorPtr &result) {
    // Check Operands Valid
    std::vector<int64_t> expectedShape = GatherOperationResultShape(params, indices, axis);
    ASSERT(result->shape.size() == expectedShape.size()) << "The size of result shape and expectedShape should be equal";
    ASSERT(result->shape.size() == result->offset.size()) << "The size of result shape and offset should be equal";
    ASSERT(params->shape.size() == params->offset.size()) << "The size of params shape and offset should be equal";
    ASSERT(indices->shape.size() == indices->offset.size()) << "The size of indices shape and offset should be equal";

    ASSERT(result->shape.size() <= NUM_VALUE_5) << "Not support shape size of result greater than 5";
    ASSERT(indices->shape.size() <= NUM_VALUE_2) << "Not support shape size of indices greater than 2";
    if (axis < 0) {
        axis += params->shape.size();
    }
    ASSERT(axis >= 0 && axis < static_cast<int>(params->shape.size()))
        << "The axis should be greater than or equal to 0 and less than shape size of params";
    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherOperation(
    Function &function, const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis) {
    const auto &paramsDynShape = params->GetDynValidShape();
    const auto &indicesDynShape = indices->GetDynValidShape();
    const int paramsRank = paramsDynShape.size();
    if (axis < 0) {
        axis += paramsRank;
        ASSERT(axis >= 0 && axis < paramsRank) << "The configuration of the axis is incorrect";
    }
    std::vector<int64_t> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);
    std::vector<SymbolicScalar> outValidShape = paramsDynShape;
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, indicesDynShape.begin(), indicesDynShape.end());
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_GATHER, {params, indices}, {result}, {outValidShape});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

Tensor Gather(const Tensor &params, const Tensor &indices, int axis) {
    DECLARE_TRACER();

    RETURN_CALL(
        GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(), indices.GetStorage(), axis);
}

Tensor TensorIndex(const Tensor &params, const Tensor &indices) {
    DECLARE_TRACER();

    // TensorIndex默认按0轴进行gather
    RETURN_CALL(
        GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(), indices.GetStorage(), 0);
}

void TiledGatherElementOperation(Function &function, const TileShape &tileShape, size_t cur, Input &paramsInput,
    Input &indicesInput, int axis, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile =
            paramsInput.tensor.GetStorage()->View(function, paramsInput.tileInfo.shape, paramsInput.tileInfo.offset);
        auto indicesTile =
            indicesInput.tensor.GetStorage()->View(function, indicesInput.tileInfo.shape, indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        Shape tmpShape({indicesTile->GetShape()[indicesTile->GetShape().size() - 1]});
        auto tmpBuffer = std::make_shared<LogicalTensor>(function, indicesTile->Datatype(), tmpShape);
        auto &op = function.AddOperation(Opcode::OP_GATHER_ELEMENT, {paramsTile, indicesTile}, {resultTile, tmpBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    // 按照resultShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur == static_cast<size_t>(axis)) {
            // params[axis]不切
            paramsInput.tileInfo.offset[cur] = 0;
            paramsInput.tileInfo.shape[cur] = paramsInput.tensor.GetShape()[cur];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor.GetShape()[cur];
            indicesInput.tileInfo.shape[cur] =
                std::min(indicesInput.tensor.GetShape()[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        } else {
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor.GetShape()[cur];
            paramsInput.tileInfo.shape[cur] =
                std::min(paramsInput.tensor.GetShape()[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor.GetShape()[cur];
            indicesInput.tileInfo.shape[cur] =
                std::min(indicesInput.tensor.GetShape()[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherElementOperation(
            function, tileShape, cur + 1, paramsInput, indicesInput, axis, result, resultTileInfo);
    }
}

void TiledGatherElementOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &params,
    const LogicalTensorPtr &indices, int axis, const LogicalTensorPtr &result) {
    // Check Operands Valid
    ASSERT(result->shape.size() == result->offset.size()) << "The size of result shape and offset should be equal";
    ASSERT(params->shape.size() == params->offset.size()) << "The size of params shape and offset should be equal";
    ASSERT(indices->shape.size() == indices->offset.size()) << "The size of indices shape and offset should be equal";

    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherElementOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherElementOperation(
    Function &function, const LogicalTensorPtr &params, const LogicalTensorPtr &indices, int axis) {
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), indices->shape);
    std::vector<std::vector<SymbolicScalar>> outValidShape;
    outValidShape.push_back(indices->GetDynValidShape());
    auto &op =
        GraphUtils::AddDynOperation(function, Opcode::OP_GATHER_ELEMENT, {params, indices}, {result}, outValidShape);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

Tensor GatherElements(const Tensor &params, const Tensor &indices, int axis) {
    DECLARE_TRACER();
    ASSERT(params.GetShape().size() == indices.GetShape().size()) << "The shape size of params and indices should be equal";
    ASSERT(axis < static_cast<int>(params.GetShape().size()) && axis >= -static_cast<int>(params.GetShape().size()))
        << "The axis out of range of params shape size";
    axis = axis < 0 ? params.GetShape().size() + axis : axis; // 支持负轴
    for (size_t i = 0; i < params.GetShape().size(); ++i) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        ASSERT(indices.GetShape()[i] <= params.GetShape()[i]) << "The shape of params and indices should be equal";
    }
    std::vector<DataType> SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_BF16};
    ASSERT(
        std::find(SUPPORT_DATATYPES.begin(), SUPPORT_DATATYPES.end(), params.GetDataType()) != SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    ASSERT(indices.GetDataType() == DT_INT32 || indices.GetDataType() == DT_INT64)
        << "The datatype of indices is incorrect";

    RETURN_CALL(GatherElementOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
        indices.GetStorage(), axis);
}

struct ScatterElementSPara {
    const LogicalTensorPtr &dstTensor;
    const LogicalTensorPtr &srcInput;
    const LogicalTensorPtr &idxInput;
    const Element &scalar;
    const int axis;
    const int scatterMode;
};

struct ScatterElementSTileInfoPara {
    TileInfo srcTileInfo;
    TileInfo idxTileInfo;
    TileInfo dstTileInfo;
};

void InnerTiledScatterElementS(size_t cur, Function &function, const TileShape &tileShape,
    const ScatterElementSPara &scatterPara, ScatterElementSTileInfoPara &scatterTileInfo) {
    const LogicalTensorPtr &dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr &srcInput = scatterPara.srcInput;
    const LogicalTensorPtr &idxInput = scatterPara.idxInput;
    const Element &scalar = scatterPara.scalar;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto srcTile = srcInput->View(function, scatterTileInfo.srcTileInfo.shape, scatterTileInfo.srcTileInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxTileInfo.shape, scatterTileInfo.idxTileInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstTileInfo.shape, scatterTileInfo.dstTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_SCATTER_ELEMENT, {srcTile, idxTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    if (vecTile[axis] < std::max(dstTensor->shape[axis], idxInput->shape[axis])) {
        ALOG_ERROR_F("the axis:%d is not allowed to be cut. tileshape:%lld dstshape:%lld idxshape:%lld", 
            axis, vecTile[axis], dstTensor->shape[axis], idxInput->shape[axis]);
    }
    ASSERT(vecTile[axis] >= dstTensor->shape[axis]) << "The axis is not supported for tile splitting";
    ASSERT(vecTile[axis] >= idxInput->shape[axis]) << "The axis is not supported for tile splitting";
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
            scatterTileInfo.idxTileInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.dstTileInfo.offset[cur] = i;
            scatterTileInfo.dstTileInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.srcTileInfo.offset[cur] = i;
            scatterTileInfo.srcTileInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
        }
        InnerTiledScatterElementS(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatterElementS(Function &function, const TileShape &tileShape, const ScatterElementSPara &scatterPara) {
    // Check Operands Valid
    ASSERT(scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The size of srcInput shape and offset should be equal";
    ASSERT(scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The size of idxInput shape and offset should be equal";
    ASSERT(scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The size of dst shape and offset should be equal";

    ScatterElementSTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
    };
    InnerTiledScatterElementS(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatterElementS(Function &function, const ScatterElementSPara &scatterPara) {
    auto &op = GraphUtils::AddDynOperation(
        function, Opcode::OP_SCATTER_ELEMENT, {scatterPara.srcInput, scatterPara.idxInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, scatterPara.scalar);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
}

static void CheckScatterElementSParamsInvalid(
    const Tensor &self, const Tensor &indices, int axis, const ScatterMode reduce) {
    ASSERT(self.GetShape().size() == indices.GetShape().size()) << "The shape size of self and indices should be equal";
    ASSERT(axis < static_cast<int>(self.GetShape().size())) << "The axis should be less than size of self shape";
    ASSERT(reduce <= ScatterMode::UNKNOWN) << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        ASSERT(indices.GetShape()[i] <= self.GetShape()[i]) << "The shape of indices and self should be equal";
    }
}

Tensor Scatter(const Tensor &self, const Tensor &indices, const Element &src, int axis, ScatterMode reduce) {
    DECLARE_TRACER();

    Tensor result(self.GetStorage()->tensor->datatype, self.GetShape());
    GraphUtils::AddDynOperation(*Program::GetInstance().GetCurrentFunction(), Opcode::OP_REGISTER_COPY,
        {self.GetStorage()}, {result.GetStorage()});

    return Scatter_(result, indices, src, axis, reduce);
}

Tensor Scatter_(const Tensor &self, const Tensor &indices, const Element &src, int axis, ScatterMode reduce) {
    DECLARE_TRACER();

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
    CheckScatterElementSParamsInvalid(operandCast, indices, axis, reduce);
    Tensor result(operandCast.GetStorage()->tensor->datatype, operandCast.GetShape());
    CALL(ScatterElementS, *Program::GetInstance().GetCurrentFunction(),
        {result.GetStorage(), operandCast.GetStorage(), indices.GetStorage(), src, axis, static_cast<int>(reduce)});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
        result.GetStorage(), orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

struct ScatterPara {
    const LogicalTensorPtr &dstTensor;
    const LogicalTensorPtr &selfInput;
    const LogicalTensorPtr &idxInput;
    const LogicalTensorPtr &srcInput;
    const int axis;
    const int scatterMode;
};

struct ScatterTileInfoPara {
    TileInfo srcInfo;
    TileInfo idxInfo;
    TileInfo dstInfo;
    TileInfo selfInfo;
};

void InnerTiledScatter(size_t cur, Function &function, const TileShape &tileShape, 
    const ScatterPara &scatterPara, ScatterTileInfoPara &scatterTileInfo) {
    const LogicalTensorPtr &dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr &selfInput = scatterPara.selfInput;
    const LogicalTensorPtr &idxInput = scatterPara.idxInput;
    const LogicalTensorPtr &srcInput = scatterPara.srcInput;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        auto selfTile = selfInput->View(function, scatterTileInfo.selfInfo.shape, scatterTileInfo.selfInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxInfo.shape, scatterTileInfo.idxInfo.offset);
        auto srcTile = srcInput->View(function, scatterTileInfo.srcInfo.shape, scatterTileInfo.srcInfo.offset);
        auto dstTile = dstTensor->View(function, scatterTileInfo.dstInfo.shape, scatterTileInfo.dstInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_SCATTER, {selfTile, idxTile, srcTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    if (vecTile[axis] < std::max(dstTensor->shape[axis], idxInput->shape[axis])) {
        ALOG_ERROR_F("the axis:%d is not allowed to be cut. tileshape:%lld dstshape:%lld idxshape:%lld", 
            axis, vecTile[axis], dstTensor->shape[axis], idxInput->shape[axis]);
    }
    ASSERT(vecTile[axis] >= dstTensor->shape[axis]) << "The axis is not supported for tile splitting";
    ASSERT(vecTile[axis] >= idxInput->shape[axis]) << "The axis is not supported for tile splitting";
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
            scatterTileInfo.idxInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur], tmpTile);
            scatterTileInfo.dstInfo.offset[cur] = i;
            scatterTileInfo.dstInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur], tmpTile);
            scatterTileInfo.srcInfo.offset[cur] = i;
            scatterTileInfo.srcInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur], tmpTile);
            scatterTileInfo.selfInfo.offset[cur] = i;
            scatterTileInfo.selfInfo.shape[cur] =
                std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur], tmpTile);
        }
        InnerTiledScatter(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatter(Function &function, const TileShape &tileShape, const ScatterPara &scatterPara) {
    // Check Operands Valid
    ASSERT(scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The shape size of srcInput and offset should be equal";
    ASSERT(scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The shape size of idxInput and offset should be equal";
    ASSERT(scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The shape size of dst and offset should be equal";
    ASSERT(scatterPara.selfInput->shape.size() == scatterPara.selfInput->offset.size())
        << "The shape size of selfInput and offset should be equal";

    ScatterTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
        TileInfo(scatterPara.selfInput->shape.size(), scatterPara.selfInput->offset.size()),
    };
    InnerTiledScatter(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatter(Function &function, const ScatterPara &scatterPara) {
    auto &op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER,
        {scatterPara.selfInput, scatterPara.idxInput, scatterPara.srcInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
}

static void CheckScatterParamsInvalid(const Tensor &self, const Tensor &indices, const Tensor &src, int axis, 
    const ScatterMode reduce) {
    ASSERT(self.GetShape().size() == indices.GetShape().size()) << "The shape size of self and indices should be equal";
    ASSERT(src.GetShape().size() == indices.GetShape().size()) << "The shape size of src and indices should be equal";
    ASSERT(axis < static_cast<int>(self.GetShape().size())) << "The axis should be less than size of self shape";
    ASSERT(reduce <= ScatterMode::UNKNOWN) << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        ASSERT(indices.GetShape()[i] <= src.GetShape()[i]) << "The shape size of src and indices should be equal";
        if (static_cast<int>(i) == axis) {
            continue;
        }
        ASSERT(indices.GetShape()[i] <= self.GetShape()[i]) << "The shape size of src and indices should be equal";
    }
}

Tensor Scatter(const Tensor &self, const Tensor &indices, const Tensor &src, int axis, ScatterMode reduce) {
    DECLARE_TRACER();

    Tensor result(self.GetStorage()->tensor->datatype, self.GetShape());
    GraphUtils::AddDynOperation(*Program::GetInstance().GetCurrentFunction(), Opcode::OP_REGISTER_COPY,
        {self.GetStorage()}, {result.GetStorage()});

    return Scatter_(result, indices, src, axis, reduce);
}

Tensor Scatter_(const Tensor &self, const Tensor &indices, const Tensor &src, int axis, ScatterMode reduce) {
    DECLARE_TRACER();

    axis = axis < 0 ? self.GetShape().size() + axis : axis;
    CheckScatterParamsInvalid(self, indices, src, axis, reduce);
    Tensor result(self.GetStorage()->tensor->datatype, self.GetShape());
    CALL(Scatter, *Program::GetInstance().GetCurrentFunction(), {result.GetStorage(), self.GetStorage(),
        indices.GetStorage(), src.GetStorage(), axis, static_cast<int>(reduce)});
    return result;
}

void TiledScatterUpdate(size_t cur, Function &function, const TileShape &tileShape, Input &srcInput, Input &indexInput,
    Input &dstInput, int axis, const LogicalTensorPtr &dst, TileInfo &dstTileInfo, std::string cacheMode,
    int blockSize) {
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto resultTile = dst->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile =
            indexInput.tensor.GetStorage()->View(function, indexInput.tileInfo.shape, indexInput.tileInfo.offset);
        auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {resultTile});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照dstShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = dst->shape[cur];
    }
    for (int i = 0; i < dst->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor.GetShape()[cur];
            if (cur <= 1) {
                indexInput.tileInfo.offset[cur] = 0;
                indexInput.tileInfo.shape[cur] = indexInput.tensor.GetShape()[cur];
            }
            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor.GetShape()[cur];
            srcInput.tileInfo.shape[cur] =
                std::min(srcInput.tensor.GetShape()[cur] - srcInput.tileInfo.offset[cur], tmpTile);
            if (cur == 0) { // only cut index first axis
                indexInput.tileInfo.offset[cur] = i % indexInput.tensor.GetShape()[cur];
                indexInput.tileInfo.shape[cur] =
                    std::min(indexInput.tensor.GetShape()[cur] - indexInput.tileInfo.offset[cur], tmpTile);
            } else {
                indexInput.tileInfo.offset[1] = 0;
                indexInput.tileInfo.shape[1] = indexInput.tensor.GetShape()[1];
            }
            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = std::min(dst->shape[cur] - dstTileInfo.offset[cur], tmpTile);
        }
        TiledScatterUpdate(
            cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo, cacheMode, blockSize);
    }
}

void TiledIndexScatterUpdate(size_t cur, Function &function, const TileShape &tileShape, Input &srcInput,
    Input &indexInput, Input &dstInput, int axis, const std::shared_ptr<LogicalTensor> &dst, TileInfo &dstTileInfo,
    std::string cacheMode, int blockSize) {
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile =
            indexInput.tensor.GetStorage()->View(function, indexInput.tileInfo.shape, indexInput.tileInfo.offset);
        auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {dst});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照srcShape进行切分
    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = srcInput.tensor.GetShape()[cur];
    }

    for (int i = 0; i < srcInput.tensor.GetShape()[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) { // asis == 1
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor.GetShape()[cur];

            int64_t indexTileLen = vecTile[0];
            indexInput.tileInfo.offset[cur] = 0;
            indexInput.tileInfo.shape[cur] =
                std::min(indexInput.tensor.GetShape()[cur] - indexInput.tileInfo.offset[0], indexTileLen);

            // indextileinfo need trans : [16,0] -> [0,16]
            indexInput.tileInfo.offset[cur] = indexInput.tileInfo.offset[0];
            indexInput.tileInfo.offset[0] = 0;

            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor.GetShape()[cur];
            srcInput.tileInfo.shape[cur] =
                std::min(srcInput.tensor.GetShape()[cur] - srcInput.tileInfo.offset[cur], tmpTile);

            indexInput.tileInfo.offset[0] = i % indexInput.tensor.GetShape()[1];
            indexInput.tileInfo.shape[0] = indexInput.tensor.GetShape()[0]; // index axis 0

            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = tmpTile;
        }
        TiledIndexScatterUpdate(
            cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo, cacheMode, blockSize);
    }
}

void TiledScatterUpdateFor2Dims(Function &function, const TileShape &tileShape, const LogicalTensorPtr &result,
    const LogicalTensorPtr &src, const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis,
    std::string cacheMode, int blockSize) {
    auto &vecTile = tileShape.GetVecTile();
    int64_t tileBS = vecTile[NUM_VALUE_0];
    int64_t tileD = vecTile[NUM_VALUE_1];
    int64_t s = index->shape[1];
    if (s == 0 || tileBS == 0) {
        ALOG_ERROR_F("error: s == 0 || tileBS == 0");
        ASSERT(s == 0 || tileBS == 0);
    }
    if ((tileBS < s && s % tileBS != 0) || (tileBS > s && tileBS % s != 0)) {
        ALOG_ERROR_F("tileshape 0 is invalid, tileshape(%d, %d)", tileBS, tileD);
    }
    ASSERT((tileBS <= s && s % tileBS == 0) || (tileBS > s && tileBS % s == 0));
    ASSERT(tileD == src->shape[NUM_VALUE_1]) << "The tileD and src shape[0] should be equal";
    int64_t tileB = CeilDiv(tileBS, s);
    int64_t tileS = tileBS < s ? tileBS : s;
    int64_t bsOffset = 0;
    for (int64_t bIdx = 0; bIdx < index->shape[0]; bIdx += tileB) {
        for (int64_t sIdx = 0; sIdx < index->shape[1]; sIdx += tileS) {
            auto indexTile = index->View(function,
                {std::min(index->shape[0] - bIdx, tileB), std::min(index->shape[1] - sIdx, tileS)}, {bIdx, sIdx});
            for (int64_t j = 0; j < src->shape[1]; j += tileD) {
                auto srcTile = src->View(function,
                    {std::min(src->shape[0] - bsOffset, tileBS), std::min(src->shape[1] - j, tileD)}, {bsOffset, j});
                auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dst}, {result});
                op.SetAttribute("axis", axis);
                op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
            }
            bsOffset += tileBS;
        }
    }
}

void TiledScatterUpdateFor4Dims(Function &function, const TileShape &tileShape, const LogicalTensorPtr &result,
    const LogicalTensorPtr &src, const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis,
    std::string cacheMode, int blockSize) {
    auto &vecTile = tileShape.GetVecTile();
    int64_t tileB = vecTile[NUM_VALUE_0];
    int64_t tileS = vecTile[NUM_VALUE_1];
    int64_t tileN = vecTile[NUM_VALUE_2];
    int64_t tileD = vecTile[NUM_VALUE_3];
    for (int64_t i = 0; i < src->shape[0]; i += tileB) {
        for (int64_t j = 0; j < src->shape[1]; j += tileS) {
            auto indexTile = index->View(
                function, {std::min(index->shape[0] - i, tileB), std::min(index->shape[1] - j, tileS)}, {i, j});
            for (int64_t n = 0; n < src->shape[2]; n += tileN) {
                for (int64_t d = 0; d < src->shape[3]; d += tileD) {
                    auto srcTile = src->View(function,
                        {std::min(src->shape[0] - i, tileB), std::min(src->shape[1] - j, tileS),
                            std::min(src->shape[2] - n, tileN), std::min(src->shape[3] - d, tileD)},
                        {i, j, n, d});
                    auto &op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dst}, {result});
                    op.SetAttribute("axis", axis);
                    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
                }
            }
        }
    }
}

void TiledScatterUpdate(Function &function, const TileShape &tileShape, const LogicalTensorPtr &result,
    const LogicalTensorPtr &src, const LogicalTensorPtr &index, const LogicalTensorPtr &dst, int axis,
    std::string cacheMode, int blockSize) {
    if (cacheMode == "PA_BSND") {
        if (src->shape.size() == NUM_VALUE_2) {
            TiledScatterUpdateFor2Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        } else if (src->shape.size() == NUM_VALUE_4) {
            TiledScatterUpdateFor4Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        } else {
            ALOG_ERROR_F("shape must be 2 or 4");
        }
        ASSERT(src->shape.size() == NUM_VALUE_2 || src->shape.size() == NUM_VALUE_4);
        return;
    }
    // Check Operands Valid
    ASSERT(result->shape.size() == result->offset.size()) << "The shape of result and offset should be equal";
    ASSERT(src->shape.size() == src->offset.size()) << "The shape of src and offset should be equal";
    ASSERT(index->shape.size() == index->offset.size()) << "The shape of index and offset should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo indexTileInfo(index->shape.size(), index->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());

    auto srcInput = Input{src, srcTileInfo};
    auto indexInput = Input{index, indexTileInfo};
    auto dstInput = Input{dst, dstTileInfo};
    auto &vecTile = tileShape.GetVecTile();
    if (axis == 1 && src->shape.size() == NUM_VALUE_2 && vecTile[1] == src->shape[1]) { // 2维切index场景
        TiledIndexScatterUpdate(
            0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo, cacheMode, blockSize);
    } else {
        TiledScatterUpdate(
            0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo, cacheMode, blockSize);
    }
}

void TensorScatterUpdate(Function &function, const LogicalTensorPtr &result, const LogicalTensorPtr &dst,
    const LogicalTensorPtr &index, const LogicalTensorPtr &src, int axis, std::string cacheMode, int blockSize) {
    std::vector<int> newOffset(src->shape.size(), 0);

    // src: ub
    // index: ub
    // dst: gm
    // result: gm
    auto &op = function.AddOperation(Opcode::OP_INDEX_OUTCAST, {src, index, dst}, {result});
    op.SetAttribute("axis", axis);
    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
}

static void CheckScatterUpdateInput(const Tensor &input) {
    if ((input.GetShape().size() == NUM_VALUE_2 &&
            (input.GetShape(NUM_VALUE_0) == NUM_VALUE_0 || input.GetShape(NUM_VALUE_1) == NUM_VALUE_0)) ||
        (input.GetShape().size() == NUM_VALUE_4 &&
            (input.GetShape(NUM_VALUE_0) == NUM_VALUE_0 || input.GetShape(NUM_VALUE_1) == NUM_VALUE_0 ||
                input.GetShape(NUM_VALUE_2) == NUM_VALUE_0 || input.GetShape(NUM_VALUE_3) == NUM_VALUE_0))) {
        ALOG_ERROR_F("input shape is zero");
    }
    ASSERT((input.GetShape().size() == NUM_VALUE_2 &&
               (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0)) ||
           (input.GetShape().size() == NUM_VALUE_4 &&
               (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0 &&
                   input.GetShape(NUM_VALUE_2) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_3) != NUM_VALUE_0)))
        << "The shape of input is invaild";
    ASSERT(input.GetShape().size() == NUM_VALUE_2 || input.GetShape().size() == NUM_VALUE_4)
        << "The shape size of input is invaild";
}

static void CheckScatterUpdateIndex(const Tensor &index) {
    if (index.GetDataType() != DT_INT64 && index.GetDataType() != DT_INT32 && index.GetDataType() != DT_INT16) {
        ALOG_ERROR_F(
            "index.GetDataType() != DT_INT64 && index.GetDataType() != DT_INT32 && index.GetDataType() != DT_INT16");
    }
    ASSERT(index.GetDataType() == DT_INT64 || index.GetDataType() == DT_INT32 || index.GetDataType() == DT_INT16)
        << "The datatype of input is not supported";
    if (index.GetShape().size() != NUM_VALUE_2 || index.GetShape(NUM_VALUE_0) == NUM_VALUE_0 ||
        index.GetShape(NUM_VALUE_1) == NUM_VALUE_0) {
        ALOG_ERROR_F("index.GetShape().size() is %d, shoud be 2", index.GetShape().size());
    }
    ASSERT(index.GetShape().size() == NUM_VALUE_2 && index.GetShape(NUM_VALUE_0) != NUM_VALUE_0 &&
           index.GetShape(NUM_VALUE_1) != NUM_VALUE_0) << "The shape of index is invaild";
}

static void CheckScatterUpdateInvalid(const Tensor &dst, const Tensor &index, const Tensor &src) {
    if (src.GetShape().size() != dst.GetShape().size()) {
        ALOG_ERROR_F("src.GetShape().size() == dst.GetShape().size()");
    }
    ASSERT(src.GetShape().size() == dst.GetShape().size()) << "The shape size of src and dst should be equal";
    CheckScatterUpdateIndex(index);
    CheckScatterUpdateInput(src);
    CheckScatterUpdateInput(dst);
}

Tensor ScatterUpdate(
    const Tensor &dst, const Tensor &index, const Tensor &src, int axis, std::string cacheMode, int chunkSize) {
    DECLARE_TRACER();

    CheckScatterUpdateInvalid(dst, index, src);
    axis = axis < 0 ? dst.GetShape().size() + axis : axis;

    Tensor result(dst.GetStorage()->tensor->datatype, dst.GetStorage()->GetShape(), "", dst.Format());
    if (std::find(dst.GetStorage()->GetShape().begin(), dst.GetStorage()->GetShape().end(), -1) !=
        dst.GetStorage()->GetShape().end()) {
        Tensor resTmp(dst.GetStorage()->tensor->datatype, dst.GetStorage()->GetDynValidShape(), "", dst.Format());
        result = resTmp;
    }

    if (cacheMode == "PA_NZ") {
        axis = 1;
        ASSERT(src.GetShape().size() == NUM_VALUE_2) << "Only support 2 dim"; // only support 2 dim

        Tensor newIndex = Reshape(index, {1, index.GetShape()[0] * index.GetShape()[1]});
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
            newIndex.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    } else {
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
            index.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    }
    return result;
}

Tensor IndexPut(const Tensor &src, std::vector<Tensor> indices, const Tensor &values) {
    DECLARE_TRACER();

    CheckScatterUpdateInput(src);
    Tensor result(src.GetStorage()->tensor->datatype, src.GetShape());
    for (auto index : indices) {
        CheckScatterUpdateIndex(index);
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), src.GetStorage(),
            index.GetStorage(), values.GetStorage(), 0, "PA_PNSD", 1);
    }

    return result;
}

template <typename T, DataType dataType>
Element GetCurStartElement(Element start, Element step, int id) {
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T curStartValue = startValue + id * stepValue;
    Element curStart(dataType, curStartValue);
    return curStart;
}

const double EPSILON = (double)1e-12;
template <typename T, DataType dataType>
int64_t GetRangeResSize(Element &start, Element &end, Element &step) {
    int64_t resultSize;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        int64_t startValue = start.GetSignedData();
        int64_t endValue = end.GetSignedData();
        int64_t stepValue = step.GetSignedData();
        if (abs(stepValue) <= 0) {
            ASSERT(false && "stepValue must not be 0");
        }
        resultSize = (endValue - startValue) % stepValue ? (endValue - startValue) / stepValue + 1 :
                                                           (endValue - startValue) / stepValue;
    } else if (dataType == DT_FP32) {
        double startValue = start.GetFloatData();
        double endValue = end.GetFloatData();
        double stepValue = step.GetFloatData();
        if (abs(stepValue) <= EPSILON) {
            ASSERT(false && "stepValue must not be 0");
        }
        resultSize = static_cast<int64_t>(std::ceil((endValue - startValue) / stepValue));
    }
    return resultSize;
}

void TiledRange(Function &function, const TileShape &tileShape, const Element start, const Element step,
    const LogicalTensorPtr &result) {
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[0]; i += vecTile[0]) {
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        int64_t curSizeValue = resultTileInfo.shape[0];
        Element curSize(DT_INT64, curSizeValue);
        Element curStart;
        if (start.GetDataType() == DT_INT32) {
            curStart = GetCurStartElement<int32_t, DT_INT32>(start, step, i);
        } else if (start.GetDataType() == DT_INT64) {
            curStart = GetCurStartElement<int64_t, DT_INT64>(start, step, i);
        } else if (start.GetDataType() == DT_FP32) {
            curStart = GetCurStartElement<float, DT_FP32>(start, step, i);
        } else {
            std::string errorMessage = "Unsupported DataType " + DataType2String(start.GetDataType());
            throw std::invalid_argument(errorMessage.c_str());
        }
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_RANGE, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "START", curStart);
        op.SetAttribute(OP_ATTR_PREFIX + "SIZE", curSize);
        op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    }
    return;
}

LogicalTensorPtr TensorRange(Function &function, LogicalTensorPtr &result, Element &start, Element &step) {
    auto &op = function.AddOperation(Opcode::OP_RANGE, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "START", start);
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    Element size(DT_INT64, result->shape[0]);
    op.SetAttribute(OP_ATTR_PREFIX + "SIZE", size);
    return result;
}

Tensor RealRange(Element &start, Element &end, Element &step) {
    DECLARE_TRACER();
    std::vector<int64_t> resTensorShape;
    int64_t resultSize;
    if (start.GetDataType() == DT_INT32) {
        resultSize = GetRangeResSize<int32_t, DT_INT32>(start, end, step);
    } else if (start.GetDataType() == DT_INT64) {
        resultSize = GetRangeResSize<int64_t, DT_INT64>(start, end, step);
    } else if (start.GetDataType() == DT_FP32) {
        resultSize = GetRangeResSize<float, DT_FP32>(start, end, step);
    } else {
        std::string errorMessage = "Unsupported DataType " + DataType2String(start.GetDataType());
        throw std::invalid_argument(errorMessage.c_str());
    }
    ASSERT(resultSize > 0 && "The positivity or negativity of the step should be aligned with the end-start");
    resTensorShape.push_back(resultSize);
    auto resTensor = Tensor(start.GetDataType(), resTensorShape);
    RETURN_CALL(Range, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), start, step);
}

bool IsDataTypeUnsupport (DataType dType){
    return dType != DT_FP32 && dType != DT_INT64 && dType != DT_INT32 && dType != DT_FP16 && dType != DT_BF16;
}

DataType GetResultDataType(const Element &start, const Element &end, const Element &step) {
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    if (IsDataTypeUnsupport(startType)) {
        std::string errorMessage = "Unsupported Start DataType " + DataType2String(startType);
        ASSERT(false && errorMessage.c_str());
    }
    if (IsDataTypeUnsupport(endType)) {
        std::string errorMessage = "Unsupported End DataType " + DataType2String(endType);
        ASSERT(false && errorMessage.c_str());
    }
    if (IsDataTypeUnsupport(stepType)) {
        std::string errorMessage = "Unsupported Step DataType " + DataType2String(stepType);
        ASSERT(false && errorMessage.c_str());
    }
    bool startIsFloat = (startType == DT_FP32 || startType == DT_FP16 || startType == DT_BF16);
    bool endIsFloat = (endType == DT_FP32 || endType == DT_FP16 || endType == DT_BF16);
    bool stepIsFloat = (stepType == DT_FP32 || stepType == DT_FP16 || stepType == DT_BF16);
    if (startIsFloat || endIsFloat || stepIsFloat) {
        return DT_FP32;
    }
    int64_t startValue = start.GetSignedData();
    int64_t endValue = end.GetSignedData();
    int64_t stepValue = step.GetSignedData();
    bool startFlag = startValue <= INT_MAX && startValue >= INT_MIN;
    bool endFlag = endValue <= INT_MAX && endValue >= INT_MIN;
    bool stepFlag = stepValue <= INT_MAX && stepValue >= INT_MIN;
    if (startFlag && endFlag && stepFlag) {
        return DT_INT32;
    }
    return DT_INT64;
}

DataType GetFloatDataType(const Element &start, const Element &end, const Element &step) {
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    if (startType == DT_FP32 || endType == DT_FP32 || stepType == DT_FP32) {
        return DT_FP32;
    }
    if (startType == DT_FP16 || endType == DT_FP16 || stepType == DT_FP16) {
        return DT_FP16;
    }
    return DT_BF16;
}

Element GetElementWithDataType(const Element &element, DataType dataType) {
    DataType elementType = element.GetDataType();
    bool elementIsFloat = (elementType == DT_FP32) || (elementType == DT_FP16) || (elementType == DT_BF16);
    if (elementIsFloat && dataType == DT_FP32) {
        return Element(dataType, element.GetFloatData());
    } else if (elementIsFloat && dataType != DT_FP32) {
        return Element(dataType, (int64_t)element.GetFloatData());
    } else if (!elementIsFloat && dataType == DT_FP32) {
        return Element(dataType, (double)element.GetSignedData());
    }
    return Element(dataType, element.GetSignedData());
}

Tensor Range(const Element &start, const Element &end, const Element &step) {
    DataType dataType = GetResultDataType(start, end, step);
    if (dataType != DT_FP32 && dataType != DT_INT32) {
        std::string errorMessage = "Unsupported Output DataType " + DataType2String(dataType);
        ASSERT(false && errorMessage.c_str());
    }
    DataType floatDataType = DT_INT32;
    if (dataType == DT_FP32) {
        floatDataType = GetFloatDataType(start, end, step);
    }
    Element realStart = GetElementWithDataType(start, dataType);
    Element realEnd = GetElementWithDataType(end, dataType);
    Element realStep = GetElementWithDataType(step, dataType);
    auto resTensor = RealRange(realStart, realEnd, realStep);
    if (floatDataType == DT_BF16) {
        return Cast(resTensor, DT_BF16);
    }
    if (floatDataType == DT_FP16) {
        return Cast(resTensor, DT_FP16);
    }
    return resTensor;
}

void IndexAddOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledIndexAdd(function, tileShape, {iOperand[0], iOperand[1], iOperand[2], oOperand[0], axis, alpha});
}

void GatherOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledGatherOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
}

void GatherElementOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledGatherElementOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
}

void ScatterElementSOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatterElementS(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], scalar, axis, scatterMode});
}

void ScatterOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatter(function, tileShape, {oOperand[0], iOperand[0], iOperand[1], iOperand[2], axis, scatterMode});
}

void IndexPutOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledScatterUpdate(function, tileShape, oOperand[0], iOperand[0], iOperand[1], iOperand[2], 0, "PA_BNSD", 1);
}

void IndexOutcastOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute("axis");
    int blockSize = op.GetIntAttribute(OpAttributeKey::panzBlockSize);
    std::string cacheMode = op.GetStringAttribute(OpAttributeKey::cacheMode);
    TiledScatterUpdate(
        function, tileShape, oOperand[0], iOperand[0], iOperand[1], iOperand[2], axis, cacheMode, blockSize);
}

void RangeOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    Element start = op.GetElementAttribute(OP_ATTR_PREFIX + "START");
    Element step = op.GetElementAttribute(OP_ATTR_PREFIX + "STEP");
    TiledRange(function, tileShape, start, step, oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, IndexAddOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_GATHER, Opcode::OP_GATHER, GatherOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, GatherElementOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ScatterElementSOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER, Opcode::OP_SCATTER, ScatterOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, IndexPutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, IndexOutcastOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RANGE, Opcode::OP_RANGE, RangeOperationTileFunc);

} // namespace npu::tile_fwk
