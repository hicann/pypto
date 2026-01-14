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
 * \file shmem_expand_funcion.cpp
 * \brief
 */

#include "distributed_expand.h"
#include "distributed_common.h"

namespace npu::tile_fwk::Distributed {
namespace {
constexpr uint16_t UB_BUFFER_BYTE_SIZE = 16 * 1024;
constexpr uint16_t DTYPE_CAST_BYTE_SIZE = 256;
constexpr uint16_t UB_ALIGIN_SIZE = 32;

template<typename AddTileOp>
void DfsTiling(const TileShape& tileShape, size_t curDim, Input& input, size_t startDim,
    uint32_t& tileIndex, Shape& dataShape, AddTileOp&& addTileOp)
{
    auto& vecTile = tileShape.GetVecTile();
    ASSERT(vecTile.size() == input.tileInfo.shape.size());
    int32_t tileRowNum = dataShape[0] / vecTile[0] + (dataShape[0] % vecTile[0] == 0 ? 0 : 1);
    int32_t tileColNum = dataShape[1] / vecTile[1] + (dataShape[1] % vecTile[1] == 0 ? 0 : 1);
    size_t inDim = input.tileInfo.shape.size();
    if (curDim == inDim) {
        ASSERT(tileIndex <= MAX_TILE_NUM) << "TileNum must be <= " << MAX_TILE_NUM << ".";
        addTileOp(tileIndex, tileRowNum, tileColNum, input);
        tileIndex++;
        return;
    }    
    for (int i = 0; i < input.tensor.GetShape()[startDim + curDim]; i += vecTile[curDim]) {
        input.tileInfo.shape[curDim] = std::min(input.tensor.GetShape()[startDim + curDim] - i, vecTile[curDim]);
        input.tileInfo.offset[curDim] = i;
        DfsTiling(tileShape, curDim + 1, input, startDim, tileIndex, dataShape, std::forward<AddTileOp>(addTileOp));
    }
}

LogicalTensorPtr View2DTile(LogicalTensorPtr& dummy,
    int32_t tileIndex, int32_t tileRowNum, int32_t tileColNum, Function& function)
{
    Shape dummyShape = dummy->shape;
    ASSERT(tileRowNum > 0 && tileColNum > 0) << "TileRowNum and tileColNum can not be 0";
    int32_t rowIndex = tileIndex / tileColNum;
    int32_t colIndex = tileIndex % tileColNum;

    int32_t baseRow = dummyShape[0] / tileRowNum;
    int32_t remRow = dummyShape[0] % tileRowNum;
    int32_t tileRowShape = rowIndex < remRow ? baseRow + 1 : baseRow;
    int32_t tileRowOffset = rowIndex < remRow ? (rowIndex * tileRowShape) : remRow * (baseRow + 1) + (rowIndex - remRow) * baseRow;

    int32_t baseCol = dummyShape[1] / tileColNum;
    int32_t remCol = dummyShape[1] % tileColNum;
    int32_t tileColShape = colIndex < remCol ? baseCol + 1 : baseCol;
    int32_t tileColOffset = colIndex < remCol ? (colIndex * tileColShape) : remCol * (baseCol + 1) + (colIndex - remCol) * baseCol;
    return dummy->View(function, {tileRowShape, tileColShape}, {tileRowOffset, tileColOffset});
}

LogicalTensorPtr View1DTile(LogicalTensorPtr& dummy,
    int32_t tileIndex, int32_t totalTileNum, Function& function)
{
    Shape dummyShape = dummy->shape;
    int32_t totalElem = dummyShape[0] * dummyShape[1];
    int32_t tileStart = tileIndex;
    int32_t tileRowStart = tileStart / dummyShape[1];
    int32_t tileColStart = tileStart % dummyShape[1];
    if (tileIndex != totalTileNum - 1) {
        return dummy->View(function, {1, 1}, {tileRowStart, tileColStart});
    }
    int32_t tileEnd = totalElem - 1;
    int32_t tileRowEnd = tileEnd / dummyShape[1];
    int32_t tileColEnd = tileEnd % dummyShape[1];
    if (tileRowStart == tileRowEnd) {
        return dummy->View(function, {1, tileColEnd - tileColStart + 1}, {tileRowStart, tileColStart});
    }
    if (tileColStart == 0) {
        return dummy->View(function, {tileRowEnd - tileRowStart + 1, dummyShape[1]}, {tileRowStart, 0});
    }
    return dummy->View(function, {1, 1}, {tileRowStart, tileColStart});
}

LogicalTensorPtr GetDummyTile(LogicalTensorPtr& dummy,
    int32_t tileIndex, int32_t tileRowNum, int32_t tileColNum, Function& function)
{
    Shape dummyShape = dummy->shape;
    int32_t totalTileNum = tileRowNum * tileColNum;
    int32_t totalDummyElem = dummyShape[0] * dummyShape[1];
    if (tileRowNum <= dummyShape[0] && tileColNum <= dummyShape[1]) {
        return View2DTile(dummy, tileIndex, tileRowNum, tileColNum, function);
    }
    if (totalTileNum <= totalDummyElem) {
        return View1DTile(dummy, tileIndex, totalTileNum, function);
    }
    return dummy;
}

bool shouldConvertDtype(DataType ubType, DataType castType)
{
    return ubType != castType;
}

Shape GetCopyBufferShape(DataType nonShmemDtype, DataType shmemDtype, Shape tileShape)
{
    const uint32_t copyNum = UB_BUFFER_BYTE_SIZE / BytesOf(nonShmemDtype);
    Shape copyShape;
    auto tileRowSize = tileShape[0];
    auto tileColSize = tileShape[1];
    if ((nonShmemDtype != shmemDtype) && (tileColSize % UB_ALIGIN_SIZE != 0)) {
        uint32_t copyColSize = copyNum > tileColSize ? tileColSize : copyNum;
        copyShape = {1, copyColSize};
    } else if (copyNum >= tileRowSize * tileColSize) {
        copyShape = {tileRowSize, tileColSize};
    } else if (copyNum >= tileColSize) {
        copyShape = {(copyNum + tileColSize - 1) / tileColSize, tileColSize};
    } else {
        copyShape = {1, copyNum};
    }
    return copyShape;
}

LogicalTensorPtr CreateAdaptiveUbTensor(Function& function, const Shape& shape, DataType ubType, DataType castType)
{
    Shape ubShape;
    int64_t ubLen = AlignUp(shape[0] * shape[1] * BytesOf(ubType), UB_ALIGIN_SIZE) / BytesOf(ubType);
    if (!shouldConvertDtype(ubType, castType)) {
        ubShape = {ubLen * 2};
    } else {
        uint64_t castSize = AlignUp(ubLen * BytesOf(castType), DTYPE_CAST_BYTE_SIZE);
        ubShape = {(ubLen + static_cast<int64_t>(castSize / BytesOf(ubType))) * 2};
    }
    return std::make_shared<LogicalTensor>(function, ubType, ubShape);
}
} // namespace

void TiledShmemPut(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 3UL) << "TiledShmemPut iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemPut oOperand size is not equal to 1";
    auto in = iOperand[0];
    auto shmemData = iOperand[1];
    auto predToken = iOperand[2]; // operand 2
    auto dummy = oOperand[0];
    TileInfo tileInfo(in->shape.size(), in->offset.size());
    auto startInput = Input{in, tileInfo};
    size_t inDim = startInput.tileInfo.shape.size();
    size_t startInDim = in->shape.size() - inDim;
    uint32_t startTileIndex = 0;
    uint32_t curDim = 0;
    DfsTiling(tileShape, curDim, startInput, startInDim, startTileIndex, in->shape, [&](uint32_t tileIndex, 
        int32_t tileRowNum, int32_t tileColNum, Input& input) {
        std::vector<int64_t> tilingShape = input.tileInfo.shape;
        std::vector<int64_t> tilingOffset = input.tileInfo.offset;
        auto inShape = in->shape;
        auto inOffset = in->offset;
        for (auto dim = 0UL, inTileDim = startInDim; dim < inDim; dim++, inTileDim++) {
            inShape[inTileDim] = tilingShape[dim];
            inOffset[inTileDim] = tilingOffset[dim];
            ASSERT(inShape[inTileDim] != 0) << "inShape view shape should not be 0";
        }
        auto inTile = input.tensor.GetStorage()->View(function, inShape, inOffset);

        auto shmemShape = shmemData->shape;
        auto shmemOffset = shmemData->offset;
        size_t startShmemDim = shmemData->shape.size() - inDim;
        for (auto dim = 0UL, shmemDim = startShmemDim; dim < inDim; dim++, shmemDim++) {
            shmemShape[shmemDim] = tilingShape[dim];
            shmemOffset[shmemDim] = tilingOffset[dim];
            ASSERT(shmemShape[shmemDim] != 0) << "shmem view shape should not be 0";
        }
        auto shmDataTile = shmemData->View(function, shmemShape, shmemOffset);

        auto dummyTile = GetDummyTile(dummy, tileIndex, tileRowNum, tileColNum, function);
        auto predTokenTile = GetDummyTile(predToken, tileIndex, tileRowNum, tileColNum, function);
        auto copyBufferShape = GetCopyBufferShape(inTile->Datatype(), shmDataTile->Datatype(), input.tileInfo.shape);
        auto ubTensor = CreateAdaptiveUbTensor(function, copyBufferShape, inTile->Datatype(), shmDataTile->Datatype());
        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_PUT, {inTile, shmDataTile, predTokenTile},
            {dummyTile, ubTensor});
        DistOpAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        distOpAttr.copyBufferShape = copyBufferShape;
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    });
}

void TiledShmemPutUB2GM(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 3UL) << "TiledShmemPut iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemPut oOperand size is not equal to 1";
    (void)tileShape;
    auto in = iOperand[0];
    auto shmemData = iOperand[1];
    auto barrierDummy = iOperand[2]; // operand 2
    auto dummy = oOperand[0];
    DistOpAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    Shape shape = in->shape;
    auto copyBufferShape = GetCopyBufferShape(in->Datatype(), shmemData->Datatype(), shape);
    auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_PUT_UB2GM, {in, shmemData, barrierDummy}, {dummy});
    distOpAttr.copyBufferShape = copyBufferShape;
    tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
}

void TiledShmemSignal(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemSignal iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemSignal oOperand size is not equal to 1";
    auto dummy = iOperand[0];
    auto shmemSignal = iOperand[1];
    auto dummyOut = oOperand[0];
    std::vector<int64_t> inShape(tileShape.GetVecTile().size());
    std::vector<int64_t> inOffset(tileShape.GetVecTile().size());
    TileInfo tileInfo(inShape, inOffset);
    auto startInput = Input{shmemSignal, tileInfo};
    size_t inDim = startInput.tileInfo.shape.size();
    size_t startDim = shmemSignal->shape.size() - inDim;
    uint32_t startTileIndex = 0;
    uint32_t curDim = 0;
    Shape dataShape = {shmemSignal->shape[shmemSignal->shape.size() - 2], shmemSignal->shape[shmemSignal->shape.size() - 1]};
    DfsTiling(tileShape, curDim, startInput, startDim, startTileIndex, dataShape, [&](uint32_t tileIndex,
        int32_t tileRowNum, int32_t tileColNum, Input& input){
        auto shmemShape = shmemSignal->shape;
        auto shmemOffset = shmemSignal->offset;
        ASSERT(shmemShape.size() > inDim && shmemOffset.size() > inDim)
            << "Size of shmemShape and shmemOffset should be more than " << inDim;

        for (auto dim = 0UL, shmemDim = startDim; dim < inDim; dim++, shmemDim++) {
            shmemShape[shmemDim] = input.tileInfo.shape[dim];
            shmemOffset[shmemDim] = input.tileInfo.offset[dim];
            ASSERT(shmemShape[shmemDim] != 0) << "shmem view shape should not be 0";
        }
        auto shmSignalTile = shmemSignal->View(function, shmemShape, shmemOffset);
        auto dummyTile = GetDummyTile(dummy, tileIndex, tileRowNum, tileColNum, function);
        auto dummyOutTile = GetDummyTile(dummyOut, tileIndex, tileRowNum, tileColNum, function);
        auto ubTensor = std::make_shared<LogicalTensor>(function, shmemSignal->Datatype(), Shape{SHMEM_SIGNAL_STRIDE});
        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_SIGNAL, {dummyTile, shmSignalTile},
            {dummyOutTile, ubTensor});
        DistOpAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::dontTouch, true);
        tileIndex++;
    });
}

void TiledShmemWaitUntil(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemWaitUntil iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemWaitUntil oOperand size is not equal to 1";
    auto dummyIn = iOperand[0];
    auto shmemSignal = iOperand[1];
    auto dummy = oOperand[0];
    int32_t vetTileSize = tileShape.GetVecTile().size();
    std::vector<int64_t> inShape(vetTileSize);
    std::vector<int64_t> inOffset(vetTileSize);
    TileInfo tileInfo(inShape, inOffset);
    auto startInput = Input{shmemSignal, tileInfo};
    size_t inDim = startInput.tileInfo.shape.size();
    size_t startDim = shmemSignal->shape.size() - inDim;
    uint32_t startTileIndex = 0;
    uint32_t curDim = 0;
    Shape dataShape = {shmemSignal->shape[shmemSignal->shape.size() - 2], shmemSignal->shape[shmemSignal->shape.size() - 1]};
    DfsTiling(tileShape, curDim, startInput, startDim, startTileIndex, dataShape, [&](uint32_t tileIndex,
        int32_t tileRowNum, int32_t tileColNum, Input& input) {
        auto shmemShape = shmemSignal->shape;
        auto shmemOffset = shmemSignal->offset;
        ASSERT(shmemShape.size() > inDim && shmemOffset.size() > inDim)
            << "Size of shmemShape and shmemOffset should be more than " << inDim;
        ASSERT(tileIndex < MAX_TILE_NUM / shmemShape[0]) << "ShmemWaitUntil tileIndex is " << tileIndex
            << ", but tileIndex should be less than " << MAX_TILE_NUM / shmemShape[0];

        for (auto dim = 0UL, shmemDim = startDim; dim < inDim; dim++, shmemDim++) {
            shmemShape[shmemDim] = input.tileInfo.shape[dim];
            shmemOffset[shmemDim] = input.tileInfo.offset[dim];
            ASSERT(shmemShape[shmemDim] != 0) << "shmem view shape should not be 0";
        }
        auto shmSignalTile = shmemSignal->View(function, shmemShape, shmemOffset);
        auto dummyInTile = GetDummyTile(dummyIn, tileIndex, tileRowNum, tileColNum, function);
        auto dummyTile = GetDummyTile(dummy, tileIndex, tileRowNum, tileColNum, function);
        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {dummyInTile, shmSignalTile},
            {dummyTile});
        DistOpAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    });
}

void TiledShmemGet(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemGet iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemGet oOperand size is not equal to 1";
    auto dummy = iOperand[0];
    auto shmemData = iOperand[1];
    auto out = oOperand[0];
    std::vector<int64_t> inShape(tileShape.GetVecTile().size());
    std::vector<int64_t> inOffset(tileShape.GetVecTile().size());
    TileInfo tileInfo(inShape, inOffset);
    auto startInput = Input{shmemData, tileInfo};
    size_t inDim = startInput.tileInfo.shape.size();
    size_t startDim = shmemData->shape.size() - inDim;
    uint32_t startTileIndex = 0;
    uint32_t curDim = 0;
    Shape dataShape = {shmemData->shape[shmemData->shape.size() - 2], shmemData->shape[shmemData->shape.size() - 1]};
    DfsTiling(tileShape, curDim, startInput, startDim, startTileIndex, dataShape, [&](uint32_t tileIndex,
        int32_t tileRowNum, int32_t tileColNum, Input& input) {
        auto shmemShape = shmemData->shape;
        auto shmemOffset = shmemData->offset;
        ASSERT(shmemShape.size() > inDim && shmemOffset.size() > inDim);

        for (auto dim = 0UL, shmemDim = startDim; dim < inDim; dim++, shmemDim++) {
            shmemShape[shmemDim] = input.tileInfo.shape[dim];
            shmemOffset[shmemDim] = input.tileInfo.offset[dim];
            ASSERT(shmemShape[shmemDim] != 0);
        }
        auto shmDataTile = shmemData->View(function, shmemShape, shmemOffset);
        auto dummyTile = GetDummyTile(dummy, tileIndex, tileRowNum, tileColNum, function);
        auto outTile = out->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto copyBufferShape = GetCopyBufferShape(out->Datatype(), shmDataTile->Datatype(), input.tileInfo.shape);
        auto ubTensor = CreateAdaptiveUbTensor(function, copyBufferShape, out->Datatype(), shmDataTile->Datatype());
        auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_GET, {dummyTile, shmDataTile}, {outTile, ubTensor});
        DistOpAttr distOpAttr;
        op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        distOpAttr.copyBufferShape = copyBufferShape;
        tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    });
}

void TiledShmemGetGM2UB(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 2UL) << "TiledShmemGet iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemGet oOperand size is not equal to 1";
    (void)tileShape;
    auto dummy = iOperand[0];
    auto shmemData = iOperand[1];
    auto out = oOperand[0];

    DistOpAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    Shape shape = out->shape;
    auto copyBufferShape = GetCopyBufferShape(out->Datatype(), shmemData->Datatype(), shape);
    auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_GET_GM2UB, {dummy, shmemData}, {out});
    distOpAttr.copyBufferShape = copyBufferShape;
    tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
}

void TiledShmemSet(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void)op;
    (void)tileShape;

    ASSERT(iOperand.size() == 2UL) << "TiledShmemSet iOperand size is not equal to 2";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemSet oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto shmemTensor = iOperand[1];
    auto out = oOperand[0];

    ASSERT(UB_BUFFER_BYTE_SIZE % REPEAT_BYTE == 0) << "UB_BUFFER_BYTE_SIZE must be a multiple of 256, but got "
        << UB_BUFFER_BYTE_SIZE;
    Shape bufferShape{static_cast<int64_t>(UB_BUFFER_BYTE_SIZE / BytesOf(shmemTensor->Datatype()))};
    auto buffer = std::make_shared<LogicalTensor>(function, shmemTensor->Datatype(), bufferShape);
    auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_SET, {predToken, shmemTensor}, {out, buffer});
    DistOpAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    distOpAttr.setBufferShape = bufferShape;
    tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
}

Shape GetReduceUbShape(int64_t rowSize, int64_t colSize, DataType dType, bool fp32Mode)
{
    Shape ubShape;
    if (fp32Mode) {
        ubShape = {rowSize * colSize +                                              // copy需要的ub大小
            rowSize * colSize * (int64_t)(BytesOf(DT_FP32) / BytesOf(dType)) +      // 存放fp32计算结果的ub大小
            (int64_t)(256 / BytesOf(dType))};                                       // fp32计算需要的额外
    } else {
        ubShape = {2 * rowSize * colSize};  // copy 和 sum 需要的ub大小
    }
    return ubShape;
}

void TiledShmemReduce(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    ASSERT(iOperand.size() == 3UL) << "TiledShmemReduce iOperand size is not equal to 3";
    ASSERT(oOperand.size() == 1UL) << "TiledShmemReduce oOperand size is not equal to 1";
    auto in = iOperand[0];
    auto shmemData = iOperand[1];
    auto dummy = iOperand[2];
    auto out = oOperand[0];

    DistOpAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    distOpAttr.extraTemplateParam = distOpAttr.fp32Mode ? "true" : "false";

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            auto inTile = in->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto dummyTile = dummy->View(function, {1, 1}, {tileIndex, 0});
            auto outTile = out->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            Shape ubShape = GetReduceUbShape(rowShape, colShape, out->Datatype(), distOpAttr.fp32Mode);
            auto ubTensor = std::make_shared<LogicalTensor>(function, out->Datatype(), ubShape);

            auto& tileOp = function.AddOperation(Opcode::OP_SHMEM_REDUCE, {inTile, shmemData, dummyTile},
                {outTile, ubTensor});
            tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledShmemBindTensor(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void)iOperand;
    (void)tileShape;
    auto &oper = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, oOperand);
    SymbolicScalar bindTensor;
    if (op.HasAttr(OpAttributeKey::bindTensor)) {
        bindTensor = op.GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
        oper.SetAttribute(OpAttributeKey::bindTensor, bindTensor);
    }
}
}   // namespace npu::tile_fwk::Distributed
