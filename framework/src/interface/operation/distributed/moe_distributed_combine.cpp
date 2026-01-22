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
 * \file moe_distributed_combine.cpp
 * \brief
 */

#include "distributed_common.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_distributed.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk::Distributed {
void MoeDistributedCombineValidateExpandX(
    const Tensor& expandX,
    const Tensor& expertScales,
    int32_t epWorldSize,
    int32_t moeExpertNum)
{
    uint32_t supportedDim = 2;
    ASSERT(expandX.GetShape().size() == supportedDim) << "The dim of \"expandX\" only supports " << supportedDim
        << ", but got " << expandX.GetShape().size();

    int32_t expandXRow = expandX.GetShape(0);
    int32_t topK = expertScales.GetShape(1);
    int32_t batchSize = expertScales.GetShape(0);
    int32_t expectedRow = std::min(topK * batchSize * epWorldSize, batchSize * moeExpertNum);
    ASSERT(expandXRow == expectedRow) << "The first axis of \"expandX\" must be the smaller value between topK (the "
        << "second axis of \"expertScales\") * batchSize (the first axis of \"expertScales\") * epWorldSize and "
        << "batchSize * moeExpertNum, topK=" << topK << ", batchSize=" << batchSize << ", epWorldSize=" << epWorldSize
        << ", moeExpertNum=" << moeExpertNum << ", the expected first axis of \"expandX\" should be " << expectedRow
        << " but got " << expandXRow;

    int32_t expandXCol = expandX.GetShape(1);
    int32_t supportedHiddenSize = 5120;
    ASSERT(expandXCol == supportedHiddenSize) << "The second axis of \"expandX\" only supports " << supportedHiddenSize
        << ", but got " << expandXCol;

    ASSERT(expandX.GetDataType() == DT_BF16) << "The data type of \"expandX\" only supports DT_BF16, but got "
        << DataType2String(expandX.GetDataType());

    ASSERT(expandX.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"expandX\" only supports ND, "
        << "but got NZ";
}

void MoeDistributedCombineValidateAssistInfoForCombine(const Tensor& assistInfoForCombine, const Tensor& expandX)
{
    uint32_t supportedDim = 2;
    ASSERT(assistInfoForCombine.GetShape().size() == supportedDim) << "The dim of \"assistInfoForCombine\" only "
        << "supports " << supportedDim << ", but got " << assistInfoForCombine.GetShape().size();

    int32_t assistInfoForCombineRow = assistInfoForCombine.GetShape(0);
    int32_t expandXRow = expandX.GetShape(0);
    ASSERT(assistInfoForCombineRow == expandXRow) << "The first axis of \"assistInfoForCombine\" must be consistent "
        << "with that of \"expandX\", but expandXRow=" << expandXRow << ", assistInfoForCombineRow="
        << assistInfoForCombineRow;

    int32_t assistInfoForCombineCol = assistInfoForCombine.GetShape(1);
    int32_t supportedAssistInfoForCombineCol = 3;
    ASSERT(assistInfoForCombineCol == supportedAssistInfoForCombineCol) << "The second axis of "
        << "\"assistInfoForCombine\" must be " << supportedAssistInfoForCombineCol << ", but got "
        << assistInfoForCombineCol;

    ASSERT(assistInfoForCombine.GetDataType() == DT_INT32) << "The data type of \"assistInfoForCombine\" only supports "
        << "DT_INT32, but got " << DataType2String(assistInfoForCombine.GetDataType());

    ASSERT(assistInfoForCombine.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of "
        << "\"assistInfoForCombine\" only supports ND, but got NZ";
}

void MoeDistributedCombineValidateRecvCounts(const Tensor& recvCounts)
{
    ASSERT(recvCounts.GetShape().size() == 1) << "The dim of \"recvCounts\" only supports 1, but got "
        << recvCounts.GetShape().size();

    int32_t recvCountsSize = recvCounts.GetShape(0);
    ASSERT(recvCountsSize == 1) << "The size of \"recvCounts\" must be 1, but recvCountsSize=" << recvCountsSize;

    ASSERT(recvCounts.GetDataType() == DT_INT32) << "The data type of \"recvCounts\" only supports DT_INT32, but got "
        << DataType2String(recvCounts.GetDataType());

    ASSERT(recvCounts.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"recvCounts\" only "
        << "supports ND, but got NZ";
}

void MoeDistributedCombineValidateExpertScales(const Tensor& expertScales)
{
    uint32_t supportedDim = 2;
    ASSERT(expertScales.GetShape().size() == supportedDim) << "The dim of \"expertScales\" only supports "
        << supportedDim << ", but got " << expertScales.GetShape().size();

    int32_t expertScalesRow = expertScales.GetShape(0);
    int32_t supportedExpertScalesRow1 = 8;
    int32_t supportedExpertScalesRow2 = 256;
    ASSERT((expertScalesRow == supportedExpertScalesRow1) || (expertScalesRow == supportedExpertScalesRow2)) << "The "
        << "first axis of \"expertScales\" only supports " << supportedExpertScalesRow1 << " or "
        << supportedExpertScalesRow2 << ", but got " << expertScalesRow;

    int32_t expertScalesCol = expertScales.GetShape(1);
    int32_t supportedExpertScalesCol = 8;
    ASSERT(expertScalesCol == supportedExpertScalesCol) << "The second axis of \"expertScales\" only supports "
        << supportedExpertScalesCol << ", but got " << expertScalesCol;

    ASSERT(expertScales.GetDataType() == DT_FP32) << "The data type of \"expertScales\" only supports DT_FP32, but got "
        << DataType2String(expertScales.GetDataType());

    ASSERT(expertScales.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"expertScales\" only "
        << "supports ND, but got NZ";
}

void MoeDistributedCombineValidateOut(const Tensor& out, const Tensor& expertScales, const Tensor& expandX)
{
    uint32_t supportedDim = 2;
    ASSERT(out.GetShape().size() == supportedDim) << "The dim of \"out\" only supports " << supportedDim << ", but got "
        << out.GetShape().size();

    int32_t outRow = out.GetShape(0);
    int32_t expertScalesRow = expertScales.GetShape(0);
    ASSERT(outRow == expertScalesRow) << "The first axis of \"out\" must be consistent with that of \"expertScales\", "
        << "but expertScalesRow=" << expertScalesRow << ", outRow=" << outRow;

    int32_t outCol = out.GetShape(1);
    int32_t expandXCol = expandX.GetShape(1);
    ASSERT(outCol == expandXCol) << "The second axis of \"out\" must be consistent with that of \"expandX\", but "
        << "expandXCol=" << expandXCol << ", outCol=" << outCol;

    ASSERT(out.GetDataType() == expandX.GetDataType()) << "The data type of \"out\" must be consistent with that of "
        << "\"expandX\",  but the data type of \"expandX\" is "<< DataType2String(expandX.GetDataType()) << " and the "
        << "data type of \"out\" is " << DataType2String(out.GetDataType());

    ASSERT(out.Format() == npu::tile_fwk::TileOpFormat::TILEOP_ND) << "The format of \"out\" only supports ND, but got "
        << "NZ";
}

void MoeDistributedCombineValidateGroup(const char* group)
{
    ASSERT(group != nullptr) << "\"group\" cannot be nullptr";
    int32_t groupLen = std::strlen(group);
    int32_t maxGroupLen = 128;
    ASSERT((groupLen >= 1) && (groupLen < maxGroupLen)) << "The length of \"group\" only supports [1, " << maxGroupLen
        << "), but got " << groupLen;
}

void MoeDistributedCombineValidateMoeEpWorldSize(int32_t epWorldSize)
{
    int32_t supportedEpWorldSize1 = 4;
    int32_t supportedEpWorldSize2 = 8;
    ASSERT((epWorldSize == supportedEpWorldSize1) || (epWorldSize == supportedEpWorldSize2)) << "epWorldSize only "
        << "supports " << supportedEpWorldSize1 << " or " << supportedEpWorldSize2 << ", but got " << epWorldSize;
}

void MoeDistributedCombineValidateMoeExpertNum(int32_t moeExpertNum)
{
    int32_t supportedMoeExpertNum = 160;
    ASSERT(moeExpertNum == supportedMoeExpertNum) << "moeExpertNum only supports " << supportedMoeExpertNum << ", but "
        << "got " << moeExpertNum;
}

void TiledMoeDistributedCombineSend(
    Function& function,
    const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand,
    const Operation& op)
{
    ASSERT(iOperand.size() == 5UL) << "TiledMoeDistributedCombineSend iOperand size is not equal to 5";
    ASSERT(oOperand.size() == 1UL) << "TiledMoeDistributedCombineSend oOperand size is not equal to 1";
    auto expandX = iOperand[0];
    auto assistInfoForCombine = iOperand[1];
    auto recvCounts = iOperand[2];
    auto shmemData = iOperand[3];
    auto shmemSignal = iOperand[4];
    auto out = oOperand[0];
    int64_t hiddenSize = expandX->shape[1];

    int64_t dataByteSize = BytesOf(expandX->Datatype());
    ASSERT(dataByteSize != 0);
    int64_t paddedColShape = AlignUp(dataByteSize * hiddenSize, COPY_BLOCK_BYTE_SIZE) / dataByteSize;
    Shape assistInfoForCombineShape = Shape{
        static_cast<int64_t>(COPY_BLOCK_BYTE_SIZE) / static_cast<int64_t>(BytesOf(DT_INT32))};
    Shape signalShape = Shape{static_cast<int64_t>(REPEAT_BYTE) / static_cast<int64_t>(BytesOf(DT_INT32))};

    DistOpAttr distOpAttr;
    op.GetAttr(OpAttributeKey::distOpAttr, distOpAttr);

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)tileIndex;

            auto expandXTile = expandX->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto dataBuffer = std::make_shared<LogicalTensor>(function, expandX->Datatype(), Shape{hiddenSize});
            auto assistInfoForCombineBuffer = std::make_shared<LogicalTensor>(function, DT_INT32, assistInfoForCombineShape);
            auto signalBuffer = std::make_shared<LogicalTensor>(function, DT_INT32, signalShape);

            auto& tileOp = function.AddOperation(Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND,
                {expandXTile, assistInfoForCombine, recvCounts, shmemData, shmemSignal},
                {out, dataBuffer, assistInfoForCombineBuffer, signalBuffer});

            distOpAttr.paddedColShape = paddedColShape;
            tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileOp.SetAttr(OpAttributeKey::dontTouch, true);
        });
}

void TiledMoeDistributedCombineReceive(
    Function& function,
    const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand,
    const Operation& op)
{
    (void)op;

    ASSERT(iOperand.size() == 4UL) << "TiledMoeDistributedCombineReceive iOperand size is not equal to 4";
    ASSERT(oOperand.size() == 1UL) << "TiledMoeDistributedCombineReceive oOperand size is not equal to 1";
    auto predToken = iOperand[0];
    auto expertScales = iOperand[1];
    auto shmemDataThisRank = iOperand[2];
    auto shmemSignalThisRank = iOperand[3];
    auto out = oOperand[0];
    int64_t topK = expertScales->shape[1];
    int64_t hiddenSize = out->shape[1];

    int64_t dataByteSize = BytesOf(out->Datatype());
    ASSERT(dataByteSize != 0);
    int64_t paddedColShape = AlignUp(dataByteSize * hiddenSize, COPY_BLOCK_BYTE_SIZE) / dataByteSize;
    int64_t floatByteSize = BytesOf(DataType::DT_FP32);
    ASSERT(floatByteSize != 0);
    int64_t floatEleNum = AlignUp(floatByteSize * paddedColShape, REPEAT_BYTE) / floatByteSize;

    DistOpAttr distOpAttr;
    distOpAttr.topK = topK;

    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void)tileIndex;

            auto shmemDataTile = shmemDataThisRank->View(function, {1, 1, topK * rowShape, colShape},
                {0, 0, topK * rowOffset, colOffset});
            auto outTile = out->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto mulFp32Buffer = std::make_shared<LogicalTensor>(function, DT_FP32, Shape{floatEleNum});
            auto sumFp32Buffer = std::make_shared<LogicalTensor>(function, DT_FP32, Shape{floatEleNum});
            auto outBuffer = std::make_shared<LogicalTensor>(function, out->Datatype(), Shape{hiddenSize});

            auto& tileOp = function.AddOperation(Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE,
                {predToken, expertScales, shmemDataTile, shmemSignalThisRank},
                {outTile, mulFp32Buffer, sumFp32Buffer, outBuffer});

            distOpAttr.paddedColShape = paddedColShape;
            distOpAttr.rowOffset = rowOffset;
            distOpAttr.rowShape = rowShape;
            tileOp.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void CreateShmemTensor(
    Tensor& shmemTensor,
    int32_t rankSize,
    int32_t hcclGroupIndex,
    DataType dataType,
    const Shape& shape,
    uint64_t memType = 0)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shmemShape{rankSize};
    shmemShape.insert(shmemShape.end(), shape.begin(), shape.end());
    auto shmemTensorInner = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    shmemTensor = shmemTensorInner;
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(shmemTensor, SlotProperty::SHMEM_TENSOR);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensorInner});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, memType,
        BytesOf(dataType) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>())));
}

Tensor MoeDistributedCombineSend(
    const Tensor& in,
    const Tensor& assistInfoForCombine,
    const Tensor& recvCounts,
    const Tensor& shmemData,
    const Tensor& shmemSignal,
    int32_t topK)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1});
    auto& op = function.AddOperation(
        Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND,
        {in.GetStorage(), assistInfoForCombine.GetStorage(), recvCounts.GetStorage(), shmemData.GetStorage(),
            shmemSignal.GetStorage()},
        {out});
    DistOpAttr distOpAttr;
    distOpAttr.topK = topK;
    op.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    return out;
}

Tensor MoeDistributedCombineReceive(
    const Tensor& predToken,
    const Tensor& expertScales,
    const Tensor& shmemData,
    const Tensor& shmemSignal)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int32_t batchSize = expertScales.GetShape(0);
    int32_t hiddenSize = shmemData.GetShape(3);
    auto out = std::make_shared<LogicalTensor>(function, shmemData.GetDataType(), Shape{batchSize, hiddenSize});
    function.AddOperation(
        Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE,
        {predToken.GetStorage(), expertScales.GetStorage(), shmemData.GetStorage(), shmemSignal.GetStorage()},
        {out});
    return out;
}

void MoeDistributedCombineValidate(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out)
{
    (void)sharedExpertNum;
    (void)sharedExpertRankNum;

    MoeDistributedCombineValidateExpandX(expandX, expertScales, epWorldSize, moeExpertNum);
    MoeDistributedCombineValidateAssistInfoForCombine(assistInfoForCombine, expandX);
    MoeDistributedCombineValidateRecvCounts(recvCounts);
    MoeDistributedCombineValidateExpertScales(expertScales);
    MoeDistributedCombineValidateOut(out, expertScales, expandX);
    MoeDistributedCombineValidateGroup(group);
    MoeDistributedCombineValidateMoeEpWorldSize(epWorldSize);
    MoeDistributedCombineValidateMoeExpertNum(moeExpertNum);
}

void MoeDistributedCombine(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out)
{
    MoeDistributedCombineValidate(expandX, assistInfoForCombine, recvCounts, expertScales, group, epWorldSize,
        moeExpertNum, sharedExpertNum, sharedExpertRankNum, out);

    int32_t batchSize = expertScales.GetShape(0);
    int32_t topK = expertScales.GetShape(1);
    int32_t hiddenSize = expandX.GetShape(1);

    int32_t shmemDataRow = topK * batchSize;
    Shape shmemDataShape = {1, shmemDataRow, hiddenSize};
    int32_t shmemSignalCol = SAME_ADDR_BYTE_SIZE / BytesOf(DataType::DT_FP32);
    Shape shmemSignalShape = {batchSize, shmemSignalCol};

    Tensor shmemData;
    Tensor shmemSignal;
    int32_t hcclGroupIndex = static_cast<int>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemTensor(shmemData, epWorldSize, hcclGroupIndex, expandX.GetDataType(), shmemDataShape);
        CreateShmemTensor(shmemSignal, epWorldSize, hcclGroupIndex, DT_INT32, shmemSignalShape);
    }
    LOOP("MoeDistributedCombine", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;

        int32_t expandXRow = expandX.GetShape(0);
        int32_t aivNum = 40;
        TileShape::Current().SetDistTile({expandXRow / aivNum, aivNum, expandXRow % aivNum}, {hiddenSize, 1, 0},
            {0, 0, 0});
        auto sendOut = MoeDistributedCombineSend(
            expandX,
            assistInfoForCombine,
            recvCounts,
            shmemData,
            shmemSignal,
            topK);

        SymbolicScalar thisRank = GetHcclRankId(group);
        auto shmemDataThisRank = View(shmemData, {1, 1, shmemDataRow, hiddenSize},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto shmemSignalThisRank = View(shmemSignal, {1, batchSize, shmemSignalCol},
            std::vector<SymbolicScalar>{thisRank, 0, 0});
        TileShape::Current().SetDistTile({1, batchSize, 0}, {hiddenSize, 1, 0}, {0, 0, 0});
        out = MoeDistributedCombineReceive(sendOut, expertScales, shmemDataThisRank, shmemSignalThisRank);
    }
}

void MoeDistributedCombineV2(const Tensor& expandX, const Tensor& assistInfoForCombine, const Tensor& recvCounts,
    const Tensor& expertScales, const char* group, uint32_t epWorldSize, uint32_t moeExpertNum,
    uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& out)
{
    MoeDistributedCombineValidate(expandX, assistInfoForCombine, recvCounts, expertScales, group, epWorldSize,
        moeExpertNum, sharedExpertNum, sharedExpertRankNum, out);

    int32_t batchSize = expertScales.GetShape(0);
    int32_t topK = expertScales.GetShape(1);
    int32_t hiddenSize = expandX.GetShape(1);

    Tensor shmemData;
    Tensor shmemSignal;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        CreateShmemData(group, epWorldSize, expandX.GetDataType(), {1, batchSize * topK, hiddenSize}, shmemData);
        CreateShmemSignal(group, shmemData, shmemSignal);
    }

    SymbolicScalar recvCountsScalar = GetTensorData(recvCounts, {0});
    Tensor sendOut(DT_INT32, {recvCountsScalar, 1}, "sendOut");
    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};
    LOOP("MoeDistributedCombineSend", FunctionType::DYNAMIC_LOOP, rowIndex, LoopRange(recvCountsScalar), unrollList) {
        SymbolicScalar rankId = GetTensorData(assistInfoForCombine, {rowIndex, 0});
        SymbolicScalar tokenId = GetTensorData(assistInfoForCombine, {rowIndex, 1});
        SymbolicScalar kOffset = GetTensorData(assistInfoForCombine, {rowIndex, 2});

        Tensor expandXTile = View(expandX, {1, hiddenSize}, {rowIndex, 0});
        Tensor shmemDataTile = View(shmemData, {1, 1, 1, hiddenSize}, {rankId, 0, topK * tokenId + kOffset, 0});
        TileShape::Current().SetVecTile({1, hiddenSize});
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        Tensor shmemPutOut = ShmemPut(expandXTile, shmemDataTile, predToken);

        Tensor shmemSignalTile = View(shmemSignal, {1, 1, 1, 1, hiddenSize}, {rankId, 0, 0, tokenId, 0});
        Tensor shmemSignalOut = ShmemSignal(shmemPutOut, shmemSignalTile, AtomicType::ADD);
        Assemble(shmemSignalOut, {rowIndex, 0}, sendOut);
    }

    SymbolicScalar thisRank = GetHcclRankId(group);
    LOOP("MoeDistributedCombineReceive", FunctionType::DYNAMIC_LOOP, tokenId, LoopRange(batchSize)) {
        Tensor shmemSignalTile = View(shmemSignal, {1, 1, 1, 1, hiddenSize}, {thisRank, 0, 0, tokenId, 0});
        TileShape::Current().SetVecTile({1, hiddenSize});
        Tensor waitUntilOut = WaitUntil(sendOut, shmemSignalTile, topK);

        TileShape::Current().SetVecTile({topK, hiddenSize});
        Tensor shmemDataTile = View(shmemData, {1, 1, topK, hiddenSize}, {thisRank, 0, topK * tokenId, 0});
        Tensor shmemGetOutFp16 = ShmemGet(waitUntilOut, shmemDataTile);

        TileShape::Current().SetVecTile({topK / 2, hiddenSize});
        Tensor shmemGetOutFp32 = npu::tile_fwk::Cast(shmemGetOutFp16, DT_FP32);

        Tensor expertScalesTile = View(expertScales, {1, topK}, {tokenId, 0});
        int64_t kTileShape = AlignUp(topK, 16);
        int64_t l0bSize = 65536;
        ASSERT((BytesOf(DT_FP32) != 0) && (kTileShape != 0));
        int64_t nTileShape = l0bSize / BytesOf(DT_FP32) / kTileShape;
        TileShape::Current().SetCubeTile({1, 1}, {kTileShape, kTileShape}, {nTileShape, nTileShape});
        Tensor matmulOutFp32 = Matrix::Matmul(DT_FP32, expertScalesTile, shmemGetOutFp32);

        Tensor matmulOutFp16 = npu::tile_fwk::Cast(matmulOutFp32, DT_BF16);

        Assemble(matmulOutFp16, {tokenId, 0}, out);
    }
}

}   // namespace npu::tile_fwk::Distributed