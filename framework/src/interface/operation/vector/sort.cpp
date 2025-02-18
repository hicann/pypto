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
 * \file sort.cpp
 * \brief
 */

#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/utils/operator_tracer.h"

namespace npu::tile_fwk {

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";
const std::string TOPK_OFFSET = OP_ATTR_PREFIX + "offset";
const std::string TOPK_VALIDBIT = OP_ATTR_PREFIX + "validBit";

void TiledBitSort(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int axis, int isLargest) {
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {resultTile});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        op.SetAttribute(TOPK_OFFSET, static_cast<int>(0));
        return;
    }
    // Jump cur axis
    if (cur == static_cast<size_t>(axis)) {
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledBitSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, isLargest);
    }
}

void TiledBitSort(Function &function, const TileShape &tileShape, const LogicalTensorPtr operand,
    const LogicalTensorPtr resOperand, int axis, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledBitSort(function, tileShape, 0, input, resOperand, resultTileInfo, axis, isLargest);
}

void TiledMrgSort(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int axis, int k, int isLargest) {
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_MRGSORT, {inputTile}, {resultTile});
        op.SetAttribute(TOPK_AXIS, axis);
        op.SetAttribute(TOPK_KVALUE, k);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }
    // Jump cur axis
    if (static_cast<int>(cur) == axis) {
        TiledMrgSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, k, isLargest);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledMrgSort(function, tileShape, cur + 1, input, result, resultTileInfo, axis, k, isLargest);
    }
}

void TiledMrgSort(Function &function, const TileShape &tileShape, const LogicalTensorPtr operand,
    const LogicalTensorPtr resOperand, int axis, int k, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledMrgSort(function, tileShape, 0, input, resOperand, resultTileInfo, axis, k, isLargest);
}

void TiledTopK(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &valueResult, const LogicalTensorPtr &indexResult, TileInfo &resultTileInfo, int axis, int k,
    int isLargest) {
    auto &vecTile = tileShape.GetVecTile();
    ASSERT(k <= vecTile[axis]) << "The k should less than or equal to" << vecTile[axis];
    if (static_cast<int>(cur) == axis) {
        auto source = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        constexpr int32_t blockSize = 32;
        constexpr int32_t kFactorSize = 4;
        constexpr int32_t kBlockFpNum = 8;
        constexpr int64_t maxNumValue = 8192;
        std::vector<int64_t> vecTileAlign = vecTile.tile;
        if (source->shape[axis] > vecTileAlign[axis] * NUM_VALUE_2) {
            int64_t sourceShapeSize = 1;
            for (const auto &num : source->shape) {
                sourceShapeSize *= num;
            }
            int64_t tileShapeSize = sourceShapeSize / source->shape[axis] * vecTileAlign[axis];
            if (sourceShapeSize < maxNumValue) {
                vecTileAlign[axis] = source->shape[axis];
            } else if (tileShapeSize < maxNumValue) {
                vecTileAlign[axis] = std::max((int64_t)blockSize, 
                    maxNumValue / (sourceShapeSize / source->shape[axis]) / blockSize * blockSize);
            }
        }
        vecTileAlign[axis] = (vecTileAlign[axis] + blockSize - 1) / blockSize * blockSize;
        auto axisTileNum = (source->shape[axis] + vecTileAlign[axis] - 1) / vecTileAlign[axis];
        auto axisBlockSizeAlign = vecTileAlign[axis];
        std::vector<int64_t> tileBitsortShape = source->shape;
        std::vector<int64_t> tileMrgsortShape = source->shape;
        std::vector<int64_t> tileSourceShape = source->shape;
        std::vector<int64_t> tileSourceOffset(tileSourceShape.size(), 0);
        std::vector<LogicalTensorPtr> sortList;
        for (int i = 0; i < input.tensor.GetShape()[axis]; i += vecTileAlign[axis]) {
            tileSourceShape[axis] = std::min(vecTileAlign[axis], source->shape[axis] - i);
            tileSourceOffset[axis] = i;
            auto inputTile = source->View(function, tileSourceShape, tileSourceOffset);
            auto tileBitsortRemain = (source->shape[axis] - i + blockSize - 1) / blockSize * blockSize;
            tileBitsortShape[axis] = std::min(axisBlockSizeAlign * kFactorSize, tileBitsortRemain * kFactorSize);
            auto bitsortTile =
                std::make_shared<LogicalTensor>(function, source->Datatype(), tileBitsortShape);
            auto &bitsortOp = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {bitsortTile});
            bitsortOp.SetAttribute(TOPK_AXIS, axis);
            bitsortOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
            bitsortOp.SetAttribute(TOPK_OFFSET, static_cast<int>(i));

            int kValue = std::min(k, static_cast<int>(tileSourceShape[axis]));
            tileMrgsortShape[axis] = (kValue + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
            auto mrgsortTile =
                std::make_shared<LogicalTensor>(function, source->Datatype(), tileMrgsortShape);
            auto &mrgsortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitsortTile}, {mrgsortTile});
            mrgsortOp.SetAttribute(TOPK_AXIS, axis);
            mrgsortOp.SetAttribute(TOPK_KVALUE, kValue);
            mrgsortOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
            sortList.push_back(mrgsortTile);
        }
        tileMrgsortShape[axis] = (k + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
        std::vector<int64_t> mrgsortResultOffset(tileMrgsortShape.size(), 0);
        std::vector<int64_t> tempShape = sortList[0]->shape;
        tempShape[axis] = NUM_VALUE_4 * tempShape[axis];
        for (size_t i = 0; i < tempShape.size() - 1; ++i) {
            tempShape[i] = 1;
        }
        tileMrgsortShape[axis] = tileMrgsortShape[axis] * axisTileNum;
        auto mrgsortBuffer = std::make_shared<LogicalTensor>(
            function, valueResult->Datatype(), tileMrgsortShape);
        std::vector<LogicalTensorPtr> tiledMrgsortList;
        for (int i = 0; i < axisTileNum; i += NUM_VALUE_4) {
            if ((axisTileNum - i) == NUM_VALUE_3) {
                auto tempTensor = std::make_shared<LogicalTensor>(
                    function, valueResult->Datatype(), tempShape);
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                    {sortList[i], sortList[i + 1], sortList[i + NUM_VALUE_2], sortList[i + NUM_VALUE_2]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_3);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            } else if ((axisTileNum - i) == NUM_VALUE_2) {
                auto tempTensor = std::make_shared<LogicalTensor>(
                    function, valueResult->Datatype(), tempShape);
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                    {sortList[i], sortList[i + 1], sortList[i + 1], sortList[i + 1]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_2);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            } else if ((axisTileNum - i) == 1) {
                tiledMrgsortList.push_back(sortList[i]);
            } else {
                auto tempTensor = std::make_shared<LogicalTensor>(
                    function, valueResult->Datatype(), tempShape);
                mrgsortResultOffset[axis] = i / NUM_VALUE_4 * sortList[0]->shape[axis];
                auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                    {sortList[i], sortList[i + 1], sortList[i + NUM_VALUE_2], sortList[i + NUM_VALUE_3]},
                    {mrgsortRepeatResult, tempTensor});
                mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_4);
                mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                tiledMrgsortList.push_back(mrgsortRepeatResult);
            }
        }
        int roundNum = 0;
        int width = 1;
        while (width < axisTileNum) {
            width = width << NUM_VALUE_2;
            roundNum++;
        }
        int tileResultIdx = 0;
        for (int i = 1; i < roundNum; ++i) {
            int tileResultNum = tiledMrgsortList.size();
            for (int j = tileResultIdx; j < tileResultNum; j += NUM_VALUE_4) {
                if ((tileResultNum - j) == NUM_VALUE_3) {
                    auto tempTensor = std::make_shared<LogicalTensor>(
                        function, valueResult->Datatype(), tempShape);
                    mrgsortResultOffset[axis] =
                        (tileResultNum + (j - tileResultIdx) / NUM_VALUE_4) * sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                        {tiledMrgsortList[j], tiledMrgsortList[j + 1], tiledMrgsortList[j + NUM_VALUE_2],
                            tiledMrgsortList[j + NUM_VALUE_2]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_3);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                } else if ((tileResultNum - j) == NUM_VALUE_2) {
                    auto tempTensor = std::make_shared<LogicalTensor>(
                        function, valueResult->Datatype(), tempShape);
                    mrgsortResultOffset[axis] =
                        (tileResultNum + (j - tileResultIdx) / NUM_VALUE_4) * sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                        {tiledMrgsortList[j], tiledMrgsortList[j + 1], tiledMrgsortList[j + 1],
                            tiledMrgsortList[j + 1]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_2);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                } else if ((tileResultNum - j) == 1) {
                    tiledMrgsortList.push_back(tiledMrgsortList[j]);
                } else {
                    auto tempTensor = std::make_shared<LogicalTensor>(
                        function, valueResult->Datatype(), tempShape);
                    mrgsortResultOffset[axis] =
                        (tileResultNum + (j - tileResultIdx) / NUM_VALUE_4) * sortList[0]->shape[axis];
                    auto mrgsortRepeatResult = mrgsortBuffer->View(function, sortList[0]->shape, mrgsortResultOffset);
                    auto &mrgSortMultiQue = function.AddOperation(Opcode::OP_TILEDMRGSORT,
                        {tiledMrgsortList[j], tiledMrgsortList[j + 1], tiledMrgsortList[j + NUM_VALUE_2],
                            tiledMrgsortList[j + NUM_VALUE_3]},
                        {mrgsortRepeatResult, tempTensor});
                    mrgSortMultiQue.SetAttribute(TOPK_VALIDBIT, NUM_VALUE_4);
                    mrgSortMultiQue.SetAttribute(TOPK_KVALUE, k);
                    tiledMrgsortList.push_back(mrgsortRepeatResult);
                }
            }
            tileResultIdx = tileResultNum;
        }
        auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &valueOp = function.AddOperation(Opcode::OP_EXTRACT, {tiledMrgsortList.back()}, {valueTile});
        valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
        valueOp.SetAttribute(TOPK_KVALUE, k);
        valueOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));

        auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &indexOp = function.AddOperation(Opcode::OP_EXTRACT, {tiledMrgsortList.back()}, {indexTile});
        indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
        indexOp.SetAttribute(TOPK_KVALUE, k);
        indexOp.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }

    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(valueResult->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledTopK(function, tileShape, cur + 1, input, valueResult, indexResult, resultTileInfo, axis, k, isLargest);
    }
}

void TiledTopK(Function &function, const TileShape &tileShape, const LogicalTensorPtr operand,
    const LogicalTensorPtr valueResult, const LogicalTensorPtr indexResult, int axis, int k, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(valueResult->shape, valueResult->offset);
    auto input = Input{operand, tileInfo};
    TiledTopK(function, tileShape, 0, input, valueResult, indexResult, resultTileInfo, axis, k, isLargest);
}

void TiledExtract(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, int maskMode, int kValue, bool isLargest) {
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_EXTRACT, {inputTile}, {resultTile});
        op.SetAttribute(EXTRACT_MASKMODE, maskMode);
        op.SetAttribute(TOPK_KVALUE, kValue);
        op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
        return;
    }

    // Jump last axis
    if (cur == input.tensor.GetShape().size() - 1) {
        TiledExtract(function, tileShape, cur + 1, input, result, resultTileInfo, maskMode, kValue, isLargest);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledExtract(function, tileShape, cur + 1, input, result, resultTileInfo, maskMode, kValue, isLargest);
    }
}

void TiledExtract(Function &function, const TileShape &tileShape, const LogicalTensorPtr operand,
    const LogicalTensorPtr resOperand, int maskMode, int kValue, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledExtract(function, tileShape, 0, input, resOperand, resultTileInfo, maskMode, kValue, isLargest);
}

void TiledArgSort(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const LogicalTensorPtr &resultDices, TileInfo &resultDicesTileInfo, int axis, int isLargest) {
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultDicesTile = resultDices->View(function, resultDicesTileInfo.shape, resultDicesTileInfo.offset);
        function.AddOperation(Opcode::OP_ARGSORT, {inputTile}, {resultDicesTile});
        return;
    }
    if (cur == static_cast<size_t>(axis)) {
        input.tileInfo.offset[cur] = 0;
        input.tileInfo.shape[cur] = input.tensor.GetShape()[cur];
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultDicesTileInfo.offset[cur] = i;
        resultDicesTileInfo.shape[cur] =
            std::min(resultDices->shape[cur] - resultDicesTileInfo.offset[cur], vecTile[cur]);
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
    }
}

void TiledArgSort(Function &function, const TileShape &tileShape, const LogicalTensorPtr operand,
    const LogicalTensorPtr resDicesOperand, int axis, int isLargest) {
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultDicesTileInfo(resDicesOperand->shape.size(), resDicesOperand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledArgSort(function, tileShape, 0, input, resDicesOperand, resultDicesTileInfo, axis, isLargest);
}

// 针对axis全排序,当前只支持axis为-1,输出为结果每32数据排序
void TensorBitsortOperation(
    Function &function, LogicalTensorPtr operand, LogicalTensorPtr resOp, int axis, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_BITSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    op.SetAttribute(TOPK_OFFSET, static_cast<int>(0));
}

// 全排序结果根据axis进行归并,当前只支持axis为-1
void TensorMrgSortOperation(
    Function &function, LogicalTensorPtr operand, LogicalTensorPtr resOp, int axis, int k, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_MRGSORT, {operand}, {resOp});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
}

void TensorExtractOperation(
    Function &function, LogicalTensorPtr operand, LogicalTensorPtr resOp, int maskMode, int k, bool isLargest) {
    auto &op = function.AddOperation(Opcode::OP_EXTRACT, {operand}, {resOp});
    op.SetAttribute(EXTRACT_MASKMODE, maskMode);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
}

void TensorTopK(Function &function, const LogicalTensorPtr &self, LogicalTensorPtr &valueResult,
    LogicalTensorPtr &indexResult, int k, int axis, bool isLargest) {
    if (!self->GetDynValidShape().empty()) {
        std::vector<SymbolicScalar> outValidShape;
        for (auto shape : self->GetDynValidShape()) {
            outValidShape.push_back(shape);
        }
        outValidShape[axis] = SymbolicScalar(k);
        valueResult->UpdateDynValidShape(outValidShape);
        indexResult->UpdateDynValidShape(outValidShape);
    }

    auto &op = function.AddOperation(Opcode::OP_TOPK, {self}, {valueResult, indexResult});
    op.SetAttribute(TOPK_AXIS, axis);
    op.SetAttribute(TOPK_KVALUE, k);
    op.SetAttribute(TOPK_ORDER, static_cast<int>(isLargest));
    return;
}

std::tuple<Tensor, Tensor> TopK(const Tensor &self, int k, int axis, bool isLargest) {
    DECLARE_TRACER();
    const auto len = static_cast<int>(self.GetShape().size());
    ASSERT(axis == (len - 1) || axis == -1) << "TopK only support last axis";
    axis = axis >= 0 ? axis : (axis + len);

    auto topkOutShape = self.GetShape();
    topkOutShape[axis] = k;
    auto valueResult = Tensor(self.GetStorage()->tensor->datatype, topkOutShape);
    auto indexResult = Tensor(DataType::DT_INT32, topkOutShape);
    CALL(TopK, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), valueResult.GetStorage(),
        indexResult.GetStorage(), k, axis, isLargest);
    return std::tie(valueResult, indexResult);
}

Tensor ArgSort(const Tensor &operand, int axis = -1, bool isLargest) {
    DECLARE_TRACER();
    const auto len = static_cast<int>(operand.GetShape().size());
    ASSERT(axis == 1 || axis == -1) << "ArgSort only support last axis";
    axis = axis >= 0 ? axis : (axis + len);
    // 首先进行全排序,全排序的输出是输入shape的2倍,另外需要在输出中增加临时空间,size变为原有的4倍
    // 需要注意,这里由于芯片限制需要对k做32元素对齐
    // [index value 2] + [index_tmp_buffer 1] / [index_value_tmp_buffer 2] * 2 = 4 * origin_size
    auto bitsortOutShape = operand.GetShape();
    bitsortOutShape[axis] = (bitsortOutShape[axis] + NUM_VALUE_31) / NUM_VALUE_32 * NUM_VALUE_32;
    bitsortOutShape[axis] *= NUM_VALUE_4; // size变成原来的4倍
    auto bitsortResTensor = Tensor(operand.GetStorage()->tensor->datatype, bitsortOutShape);
    CALL(BitsortOperation, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage(),
        bitsortResTensor.GetStorage(), axis, isLargest);

    auto k = operand.GetShape()[axis];
    // 归并排序,输入为全排序的结果,输出为原始输入shape的2倍
    auto mrgSortOutShape = operand.GetShape();
    constexpr int32_t MRG_SORT_TIMES_2 = 2;
    mrgSortOutShape[axis] = k * MRG_SORT_TIMES_2;
    auto mrgsortResultTensor = Tensor(operand.GetStorage()->tensor->datatype, mrgSortOutShape);
    CALL(MrgSortOperation, *Program::GetInstance().GetCurrentFunction(), bitsortResTensor.GetStorage(),
        mrgsortResultTensor.GetStorage(), axis, k, isLargest);

    // // index拆分
    auto topkOutShape = operand.GetShape();
    topkOutShape[axis] = k;

    // value 拆分
    auto resIndicesTensor = Tensor(operand.GetStorage()->tensor->datatype, topkOutShape);
    CALL(ExtractOperation, *Program::GetInstance().GetCurrentFunction(), mrgsortResultTensor.GetStorage(),
        resIndicesTensor.GetStorage(), 1, k, isLargest);
    return resIndicesTensor;
}

void BitSortOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    int axis = op.GetIntAttribute(TOPK_AXIS);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    TiledBitSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest);
}

void MrgSortOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    int axis = op.GetIntAttribute(TOPK_AXIS);
    int kValue = op.GetIntAttribute(TOPK_KVALUE);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    TiledMrgSort(function, tileShape, iOperand[0], oOperand[0], axis, kValue, isLargest);
}

void ArgSortOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    int axis = op.GetIntAttribute("axis");
    int isLargest = op.GetIntAttribute("order");
    TiledArgSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest);
}

void ExtractOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    int maskMode = op.GetIntAttribute(EXTRACT_MASKMODE);
    int kValue = op.GetIntAttribute(TOPK_KVALUE);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    TiledExtract(function, tileShape, iOperand[0], oOperand[0], maskMode, kValue, isLargest);
}

void TopkOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    int axis = op.GetIntAttribute(TOPK_AXIS);
    int kValue = op.GetIntAttribute(TOPK_KVALUE);
    int isLargest = op.GetIntAttribute(TOPK_ORDER);
    TiledTopK(function, tileShape, iOperand[0], oOperand[0], oOperand[1], axis, kValue, isLargest);
}

REGISTER_OPERATION_TILED_FUNC(OP_BITSORT, Opcode::OP_BITSORT, BitSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_MRGSORT, Opcode::OP_MRGSORT, MrgSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ARGSORT, Opcode::OP_ARGSORT, ArgSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXTRACT, Opcode::OP_EXTRACT, ExtractOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TOPK, Opcode::OP_TOPK, TopkOperationTileFunc);

} // namespace npu::tile_fwk