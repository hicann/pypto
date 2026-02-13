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
 * \file moe_dispatch.cpp
 * \brief
 */
#include <functional>
#include <memory>
#include <vector>
#include "interface/operation/operation.h"
#include "tilefwk/tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "distributed_common.h"
#include "tilefwk/symbolic_distributed.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
namespace Distributed {
void TiledDispatchFFNSched(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto syncTensor = iOperand[DIST_INDEX_ZERO];
    auto shmemFlag = iOperand[DIST_INDEX_ONE];
    auto recvTokenCntOut = oOperand[DIST_INDEX_ZERO];
    int flagColSize = shmemFlag->GetShape()[3];
    std::string hcclGroupIndex;
    std::vector<int64_t> bufferShape;
    int32_t sharedExpertNum = 0;
    int64_t expertNumPerRank;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    op.GetAttr("dispatchBufferSize", bufferShape);
    op.GetAttr("expertNumPerRank", expertNumPerRank);

    const auto &tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];
    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape :tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemFlagTile = shmemFlag->View(function, {1, 1, rankShape, flagColSize}, 
                {0, expertIndex, rankOffset, 0});
            auto &opr = function.AddOperation(Opcode::OP_FFN_SCHED, {syncTensor, shmemFlagTile}, 
                {recvTokenCntOut, bufferTensor});
            std::string extraParam = std::to_string(tileIndex) + ", " + hcclGroupIndex + ", " + 
                std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " + 
                std::to_string(rankShape) + ", " + std::to_string(expertNumPerRank);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchFFNCombineInfo(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemData = iOperand[DIST_INDEX_ONE];
    auto shmemFlag = iOperand[DIST_INDEX_TWO];
    auto combineInfo = oOperand[DIST_INDEX_ZERO];

    int32_t shmemDataLength = shmemData->GetShape()[3];
    Shape combineInfoBufferShape = {combineInfo->GetShape()[0] + 32};
    std::string hcclGroupIndex;
    std::vector<int64_t> bufferShape;
    std::string axisH;
    std::string batchSize;
    int32_t sharedExpertNum = 0;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    op.GetAttr("dispatchBufferSize", bufferShape);
    op.GetAttr("hiddenSize", axisH);
    op.GetAttr("tokenBatchSize", batchSize);

    const auto &tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];

    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape :tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferCombineInfo = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemDataTile = shmemData->View(function, {1, rankShape, 1, shmemDataLength}, 
                {0, rankOffset, expertIndex, 0});
            auto &opr = function.AddOperation(Opcode::OP_FFN_COMBINEINFO, {shmemDataTile, shmemFlag, recvTokenCntOut}, 
                {combineInfo, bufferCombineInfo});
            std::string extraParam = std::to_string(tileIndex) + ", " + hcclGroupIndex + ", " +
                std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                std::to_string(rankShape) + ", " + axisH + ", " + batchSize + ", " +
                std::to_string(combineInfo->GetShape()[0]);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchFFNBatching(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemData = iOperand[DIST_INDEX_ONE];
    auto shmemFlag = iOperand[DIST_INDEX_TWO];
    auto expandX = oOperand[DIST_INDEX_ZERO];
    auto validCnt = oOperand[DIST_INDEX_ONE];

    int32_t shmemDataLength = shmemData->GetShape()[3];
    std::string groupIndex;
    std::vector<int64_t> bufferShape;
    std::string axisH;
    std::string batchSize;
    int32_t sharedExpertNum = 0;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", groupIndex);
    op.GetAttr("dispatchBufferSize", bufferShape);
    op.GetAttr("hiddenSize", axisH);
    op.GetAttr("tokenBatchSize", batchSize);

    const auto &tileRank = tileShape.GetDistTileRank();
    int32_t totalTileNum = GetTotalTileNum(tileRank) * static_cast<int32_t>(expertNumPerRank);
    const int32_t tileRankShape = tileRank[DIST_HEAD_SHAPE];
    const int32_t tileRankCnt = tileRank[DIST_HEAD_COUNT] + (tileRank[DIST_TAIL_SHAPE] == 0 ? 0 : 1);
    const int32_t tailRankShape = tileRank[DIST_TAIL_SHAPE];

    int32_t tileIndex = 0;
    for (int expertIndex = 0; expertIndex < expertNumPerRank; ++expertIndex) {
        for (int rankIndex = 0; rankIndex < tileRankCnt; ++rankIndex) {
            int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == tileRankCnt - 1) ? tailRankShape :tileRankShape);
            int32_t rankOffset = rankIndex * tileRankShape;
            auto bufferTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
            auto shmemDataTile = shmemData->View(function, {1, rankShape, 1, shmemDataLength}, 
                {0, rankOffset, expertIndex, 0});
            auto &opr = function.AddOperation(Opcode::OP_FFN_BATCHING, {shmemDataTile, shmemFlag, recvTokenCntOut}, 
                {expandX, validCnt, bufferTensor});
            std::string extraParam = std::to_string(tileIndex) + ", " + groupIndex + ", " +
                std::to_string(sharedExpertNum) + ", " + std::to_string(totalTileNum) + ", " +
                std::to_string(rankShape) + ", " + axisH + ", " + batchSize + ", " +
                std::to_string(expandX->GetShape()[0]);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            opr.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
            tileIndex++;
        }
    }
}

void TiledDispatchFFNValidCnt(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    (void) op;
    auto recvTokenCntOut = iOperand[DIST_INDEX_ZERO];
    auto shmemFlag = iOperand[DIST_INDEX_ONE];
    auto validCnt = oOperand[DIST_INDEX_ZERO];

    int32_t flagColSize = shmemFlag->GetShape()[3];
    int32_t rankSize = shmemFlag->GetShape()[0];

    const auto& tileExpert = tileShape.GetDistTileRank();
    int32_t tileExpertShape = tileExpert[0];
    int32_t expertCount = tileExpert[1] + (tileExpert[2] == 0 ? 0 : 1);
    Shape bufferShape {shmemFlag->shape[0] * expertCount};

    for (int32_t expertIndex = 0; expertIndex < expertCount; ++expertIndex) {
        int32_t expertShape = ((tileExpert[2] != 0) && (expertIndex == expertCount - 1)) ? tileExpert[2] : tileExpert[0];
        int32_t expertOffset = expertIndex * tileExpertShape;
        auto validCntBuffer = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, bufferShape);
        auto shmemFlagTile = shmemFlag->View(function, {1, expertShape, rankSize, flagColSize},
            {0, expertOffset, 0, 0});
        auto &tileop = function.AddOperation(Opcode::OP_FFN_VALIDCNT, {recvTokenCntOut, shmemFlagTile},
            {validCnt, validCntBuffer});
        std::string extraParam = std::to_string(expertShape);
        DistOpAttr distOpAttr;
        distOpAttr.extraTemplateParam = extraParam;
        tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
    }
}

Tensor DispatchFFNValidCnt(const Tensor& recvTokenCntOut, const Tensor& shmemFlag, const MoeConfig& moeConfig)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape validCntShape = {moeConfig.expertNumPerRank, 1};
    auto validCntPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, validCntShape);
    auto &oper = function.AddOperation(Opcode::OP_FFN_VALIDCNT, {recvTokenCntOut.GetStorage(), shmemFlag.GetStorage()}, {validCntPtr});
    (void)oper;
    return validCntPtr;
}

Tensor DispatchFFNCombineInfo(const char *group, const Tensor &tokenTensor,
    const Tensor &recvTokenCntOut, const Tensor &shmemData, const Tensor &shmemFlag,
    int32_t expandXRow, int32_t ffnTileNum, const MoeConfig &moeConfig)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape combineInfoShape = {expandXRow, 3};
    auto combineInfoPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, combineInfoShape);
    auto &oper = function.AddOperation(Opcode::OP_FFN_COMBINEINFO, {recvTokenCntOut.GetStorage(), shmemData.GetStorage(),
        shmemFlag.GetStorage()}, {combineInfoPtr});
    int tempBufSize = AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 32, 256) + 256 +
        AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 4, 32) + 512; // tempBufSize = recvTokenCnt数 + 存储mask + recvTokenCnt的int数 + 数据搬运
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    const std::vector<int64_t> bufferShape {tempBufSize};
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("hiddenSize", std::to_string(tokenTensor.GetShape()[1]));
    oper.SetAttr("tokenBatchSize", std::to_string(tokenTensor.GetShape()[0]));
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return combineInfoPtr;
}

Tensor DispatchFFNBatching(const char *group, const Tensor &tokenTensor, 
    const Tensor &recvTokenCntOut, const Tensor &shmemData, const Tensor &shmemFlag, 
    int32_t expandXRow, int32_t ffnTileNum, const MoeConfig &moeConfig)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape validCntShape = {moeConfig.expertNumPerRank, 1};
    auto validCntPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, validCntShape);
    Shape expandXShape = {expandXRow, tokenTensor.GetShape()[1]};
    auto expandXPtr = std::make_shared<LogicalTensor>(function, tokenTensor.GetDataType(), expandXShape);
    auto &oper = function.AddOperation(Opcode::OP_FFN_BATCHING, {recvTokenCntOut.GetStorage(), shmemData.GetStorage(),
        shmemFlag.GetStorage()}, {expandXPtr, validCntPtr});
    int cumSumBuffer = AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 32, 256)
        + 256 + AlignUp(moeConfig.expertNumPerRank * ffnTileNum * 4, 32) + 512; // tempBufSize = recvTokenCnt数 + 存储mask + recvTokenCnt的int数 + 数据搬运
    int tokenCopyBuffer = tokenTensor.GetShape(1);
    int tempBufSize = (cumSumBuffer < tokenCopyBuffer) ? tokenCopyBuffer : cumSumBuffer;
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    const std::vector<int64_t> bufferShape {tempBufSize};
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("hiddenSize", std::to_string(tokenTensor.GetShape()[1]));
    oper.SetAttr("tokenBatchSize", std::to_string(tokenTensor.GetShape()[0]));
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return expandXPtr;
}

Tensor DispatchFFNSched(const char *group, const Tensor &flagDummy, Tensor &shmemFlag, const MoeConfig &moeConfig, int32_t ffnTileCnt)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    int32_t totalTileNum = moeConfig.routedExpertNum * ffnTileCnt;
    Shape shape = {totalTileNum, 512}; // 每个flag_count预留512个int存储
    auto recvTokenCntOutPtr = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto &oper = function.AddOperation(Opcode::OP_FFN_SCHED, {flagDummy.GetStorage(), shmemFlag.GetStorage()},
        {recvTokenCntOutPtr});
    int32_t moeOpProcessRankSize = ffnTileCnt;
    int32_t maxProcessRankSize = moeOpProcessRankSize;
    int tempBufSize = maxProcessRankSize * 32 + 256 + AlignUp(maxProcessRankSize * 4, 256);
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    const std::vector<int64_t> bufferShape {tempBufSize / 8, 8};
    oper.SetAttr("dispatchBufferSize", bufferShape);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return recvTokenCntOutPtr;
}

std::vector<int64_t> GetCommBufferSize(const std::shared_ptr<LogicalTensor> &tokenTensor)
{
    const int64_t hOutSize = tokenTensor->shape[1] * BytesOf(tokenTensor->Datatype());
    constexpr int64_t scaleParamPad = 512;
    const int64_t hCommuSize = AlignUp(hOutSize, 512) + scaleParamPad;
    return {1, static_cast<int64_t>(hCommuSize / BytesOf(tokenTensor->Datatype()))};
}

void TiledSendToRoutingExpert(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemData = iOperand[DIST_INDEX_ZERO];
    auto tokenTensor = iOperand[DIST_INDEX_ONE];
    auto expertTable = iOperand[DIST_INDEX_TWO];
    auto syncTensor = oOperand[DIST_INDEX_ZERO];
    std::string hcclGroupIndex;
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void) tileIndex;
            auto expertTableTile = expertTable->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto expertBufferUb = std::make_shared<LogicalTensor>(function, expertTable->Datatype(),
                std::vector<int64_t>{1, expertTable->shape[0] * expertTable->shape[1]});
            auto expertBuffer = std::make_shared<LogicalTensor>(function, expertTable->Datatype(),
                std::vector<int64_t>{1, expertTable->shape[0] * expertTable->shape[1] *
                (static_cast<int64_t>(sizeof(int32_t)) + 1)});
            auto tokenBuffer = std::make_shared<LogicalTensor>(function, tokenTensor->Datatype(), 
                GetCommBufferSize(tokenTensor));
            auto &tileop = function.AddOperation(Opcode::OP_SEND_TO_ROUTING_EXPERT, {tokenTensor, shmemData, 
                expertTableTile}, {syncTensor, tokenBuffer, expertBufferUb, expertBuffer});
            std::string extraParam = std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(rowOffset) +
                ", " + std::to_string(colOffset) + ", " + std::to_string(rowShape) +
                ", " + std::to_string(colShape) + ", " + hcclGroupIndex;
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledSendToSharedExpert(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemData = iOperand[DIST_INDEX_ZERO];
    auto tokenTensor = iOperand[DIST_INDEX_ONE];
    auto syncTensor = oOperand[DIST_INDEX_ZERO];
    (void) oOperand;
    std::string hcclGroupIndex;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void) tileIndex;
            Shape shape = {rowShape, colShape};
            auto tokenTensorTile = tokenTensor->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto tokenBuffer = std::make_shared<LogicalTensor>(function, tokenTensor->Datatype(), 
                GetCommBufferSize(tokenTensor));
            auto &tileop = function.AddOperation(Opcode::OP_SEND_TO_SHARED_EXPERT, {tokenTensorTile, shmemData},
                {syncTensor, tokenBuffer});
            std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
                std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(rowShape) + ", " + hcclGroupIndex;
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledCopyToLocalExpert(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto tokenTensor = iOperand[DIST_INDEX_ZERO];
    auto expandX = oOperand[DIST_INDEX_ZERO];
    auto syncTensor = oOperand[DIST_INDEX_ONE];
    (void) op;
    CreateTileOp(tileShape,
        [&](int32_t tileIndex, int32_t rowOffset, int32_t colOffset, int32_t rowShape, int32_t colShape) {
            (void) tileIndex;
            auto tokenTensorTile = tokenTensor->View(function, {rowShape, colShape}, {rowOffset, colOffset});
            auto tokenBuffer = std::make_shared<LogicalTensor>(function, tokenTensor->Datatype(), 
                GetCommBufferSize(tokenTensor));
            auto &tileop = function.AddOperation(Opcode::OP_COPY_TO_LOCAL_EXPERT, {tokenTensorTile},
                {expandX, syncTensor, tokenBuffer});
            std::string extraParam = std::to_string(tokenTensor->shape[0]) + ", " +
                std::to_string(tokenTensor->shape[1]) + ", " + std::to_string(rowShape);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        });
}

void TiledDispatchSetFlag(Function& function, const TileShape& tileShape,
    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op)
{
    auto shmemFlag = iOperand[DIST_INDEX_ZERO];
    auto syncTensor = iOperand[DIST_INDEX_ONE];
    auto tokenExpertTable = iOperand[DIST_INDEX_TWO];
    auto syncDummy = oOperand[DIST_INDEX_ZERO];
    int flagColSize = shmemFlag->GetShape()[3];
    std::string hcclGroupIndex;
    op.GetAttr("hcclGroupIndex", hcclGroupIndex);
    int64_t expertNumPerRank;
    op.GetAttr("expertNumPerRank", expertNumPerRank);

    const auto &tileExpert = tileShape.GetDistTileRank();
    const auto &tileRank = tileShape.GetDistTileCol();
    int32_t tileRankShape = tileRank[0];
    int32_t tileExpertShape = tileExpert[0];
    int32_t rankCount = tileRank[1] + (tileRank[2] == 0 ? 0 : 1);
    int32_t expertCount = tileExpert[1] + (tileExpert[2] == 0 ? 0 : 1);

    for (int32_t rankIndex = 0; rankIndex < rankCount; ++rankIndex) {
        int32_t rankShape = ((tileRank[2] != 0) && (rankIndex == rankCount - 1)) ? tileRank[2] : tileRank[0];
        for (int32_t expertIndex = 0; expertIndex < expertCount; ++expertIndex) {
            int32_t expertShape = ((tileExpert[2] != 0) && (expertIndex == expertCount - 1)) ?
                tileExpert[2] : tileExpert[0];
            int32_t rankOffset = rankIndex * tileRankShape;
            int32_t expertOffset = expertIndex * tileExpertShape;
            auto statusTensor = std::make_shared<LogicalTensor>(function, tokenExpertTable->Datatype(),
                std::vector<int64_t>{1, expertNumPerRank * 16 + 32}); // 每个expert预留16B缓存flag跟count,最后一个expert后预留32位
            auto expertBufferUb = std::make_shared<LogicalTensor>(function, tokenExpertTable->Datatype(),
                std::vector<int64_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1]});
            auto expertBuffer = std::make_shared<LogicalTensor>(function, tokenExpertTable->Datatype(),
                std::vector<int64_t>{1, tokenExpertTable->shape[0] * tokenExpertTable->shape[1] *
                (static_cast<int64_t>(sizeof(int32_t)) + 1)});
            auto shmemFlagTile = shmemFlag->View(function, {rankShape, expertShape, 1, flagColSize}, 
                {rankOffset, expertOffset, 0, 0});
            auto &tileop = function.AddOperation(Opcode::OP_DISPATCH_SET_FLAG, {tokenExpertTable, shmemFlagTile, 
                syncTensor}, {syncDummy, statusTensor, expertBufferUb, expertBuffer}); 
            std::string extraParam = std::to_string(tokenExpertTable->shape[0]) + ", " +
                std::to_string(tokenExpertTable->shape[1]) + ", " + hcclGroupIndex + ", " +
                std::to_string(expertShape) + ", " + std::to_string(rankShape);
            DistOpAttr distOpAttr;
            distOpAttr.extraTemplateParam = extraParam;
            tileop.SetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        }
    }
}

Tensor SendToRoutingExpert(const Tensor &shmemData, const Tensor &tokenTensor,
    const Tensor &tokenExpertTable, const char *group, const MoeConfig &moeConfig)
{
    Shape shape{1, 1};
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto syncTensor = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto &oper = function.AddOperation(Opcode::OP_SEND_TO_ROUTING_EXPERT, {shmemData.GetStorage(),
        tokenTensor.GetStorage(), tokenExpertTable.GetStorage()}, {syncTensor});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));    
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return syncTensor;
}

void SendToSharedExpert(const Tensor &shmemData, const Tensor &tokenTensor, 
    const Tensor &syncTensor, const char *group)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto &oper = function.AddOperation(Opcode::OP_SEND_TO_SHARED_EXPERT, {shmemData.GetStorage(), 
        tokenTensor.GetStorage()},
        {syncTensor.GetStorage()});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
}

Tensor DispatchSetFlag(Tensor &shmemFlag, const Tensor &tokenExpertTable, const Tensor &syncTensor,
    const char *group, const MoeConfig &moeConfig)
{
    Shape shape = {1, 1};
    auto &function = *Program::GetInstance().GetCurrentFunction();
    auto syncDummy = std::make_shared<LogicalTensor>(function, DataType::DT_INT32, shape);
    auto &oper = function.AddOperation(Opcode::OP_DISPATCH_SET_FLAG, {shmemFlag.GetStorage(), syncTensor.GetStorage(), 
        tokenExpertTable.GetStorage()}, {syncDummy});
    std::string hcclGroupIndex = std::to_string(CommGroupRecorder::GetInstance().Input(std::string(group)));
    oper.SetAttr("hcclGroupIndex", hcclGroupIndex);
    oper.SetAttr("expertNumPerRank", static_cast<int64_t>(moeConfig.expertNumPerRank));
    return syncDummy;
}

Tensor CopyToLocalExpert(const Tensor &tokenTensor, const Tensor &syncTensor, const MoeConfig &moeConfig)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape expandXShape = {tokenTensor.GetShape()[0] * moeConfig.routedExpertNum, tokenTensor.GetShape()[1]};
    auto expandXPtr = std::make_shared<LogicalTensor>(function, tokenTensor.GetDataType(), expandXShape);
    auto &oper = function.AddOperation(Opcode::OP_COPY_TO_LOCAL_EXPERT, {tokenTensor.GetStorage()}, 
        {expandXPtr, syncTensor.GetStorage()});
    (void) oper;
    return expandXPtr;
}

std::tuple<int32_t, int32_t, int32_t> GetFFNTileParam(const MoeConfig &moeConfig)
{
    int32_t tileRankCnt = moeConfig.rankNum > FFN_TILE_SIZE ? FFN_TILE_SIZE : moeConfig.rankNum;
    int32_t tileNum = tileRankCnt == FFN_TILE_SIZE ? moeConfig.rankNum / FFN_TILE_SIZE : 1;
    int32_t tailNum = tileNum == 1 ? 0 : (moeConfig.rankNum % FFN_TILE_SIZE == 0 ? 0 : 1);
    return {tileRankCnt, tileNum, tailNum};
}

void MoeDispatchValidateV1(const Tensor &tokenTensor, const Tensor &tokenExpertTable, Tensor &expandX,
    Tensor &validCnt, Tensor &combineInfo, const char *group, const MoeConfig &moeConfig)
{
    std::string assertResult;
    ASSERT(checkValidConfig(moeConfig, assertResult)) << assertResult;
    ASSERT(group != nullptr) << "MoeDispatch constraint violated: group name can't be nullptr.";
    ASSERT(group[0] != '\0') << "MoeDispatch constraint violated: group name is not valid.";
    ASSERT(strnlen(group, 128) < 128) << "MoeDispatch constraint violated: group name max size must be 128.";
    ASSERT(checkValidInput(tokenTensor, 2, DataType::DT_BF16, 8, 5120, assertResult)) << assertResult; // 当前仅支持shape:8,5120
    int32_t expandXRow = std::min(static_cast<int32_t>(tokenTensor.GetShape(0)) *
        static_cast<int32_t>(tokenExpertTable.GetShape(1)) * moeConfig.rankNum, static_cast<int32_t>(tokenTensor.GetShape(0)) * moeConfig.routedExpertNum);
    ASSERT(checkValidInput(tokenExpertTable, 2, DataType::DT_INT32, 8, 8, assertResult)) << assertResult; // 当前仅支持shape:8,8
    ASSERT(checkValidInput(validCnt, 1, DataType::DT_INT32, moeConfig.expertNumPerRank, 1, assertResult)) << assertResult;
    ASSERT(checkValidInput(expandX, 2, DataType::DT_BF16, expandXRow, 5120, assertResult)) << assertResult; // 当前仅支持hiddenSize:5120
    ASSERT(checkValidInput(combineInfo, 2, DataType::DT_INT32, expandXRow, 3, assertResult)) << assertResult; // comBineInfo固定hiddenSize:3
}

void CreateShmemDispatchLoop(Tensor& shmemData, Tensor& shmemFlag, const char *group,
    const MoeConfig &moeConfig, int32_t shmemDataCol, int32_t flagCol, DataType tokenTendorDtype)
{
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void) index;
        Shape shmemDataShape = {moeConfig.rankNum, moeConfig.expertNumPerRank, shmemDataCol};
        Shape shmemFlagShape = {moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol};
        CreateShmemData(group, moeConfig.rankNum, tokenTendorDtype, shmemDataShape, shmemData, 0);
        CreateShmemData(group, moeConfig.rankNum, DT_INT32, shmemFlagShape, shmemFlag, 0);
    }
}

void MoeDistributedDispatch(const Tensor &tokenTensor, const Tensor &tokenExpertTable, Tensor &expandX,
    Tensor &validCnt, Tensor &combineInfo, const char *group, const MoeConfig &moeConfig)
{
    MoeDispatchValidateV1(tokenTensor, tokenExpertTable, expandX, validCnt, combineInfo, group, moeConfig);
    SymbolicScalar thisRank = GetHcclRankId(group);
    int batchSize = tokenTensor.GetShape(0);
    int hiddenSize = tokenTensor.GetShape(1);
    int topK = tokenExpertTable.GetShape(1);
    int shmemDataLength = AlignUp(hiddenSize, 512) + 512;
    int flagRow = 1;
    int flagCol = 128;
    int shmemDataCol = shmemDataLength * batchSize;
    Tensor shmemData;
    Tensor shmemFlag;
    CreateShmemDispatchLoop(shmemData, shmemFlag, group, moeConfig, shmemDataCol, flagCol, tokenTensor.GetDataType());
    LOOP("L0", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void) index;
        TileShape::Current().SetDistTile(
            {1, batchSize, 0},
            {topK, 1, 0},
            {moeConfig.rankNum, 1, 0});
        Tensor syncTensor = SendToRoutingExpert(shmemData, tokenTensor, tokenExpertTable, group, moeConfig);
        TileShape::Current().SetDistTile(
            {flagRow, 1, 0},
            {moeConfig.rankNum, 1, 0},
            {1, moeConfig.expertNumPerRank, 0});
        auto localShmemFlag = View(shmemFlag, {moeConfig.rankNum, moeConfig.expertNumPerRank, 1, flagCol}, 
            {0, 0, thisRank, 0});
        Tensor flagDummy = DispatchSetFlag(localShmemFlag, tokenExpertTable, syncTensor, group, moeConfig);
        auto [ffnTileCnt, ffnTileNum, ffnTailNum] = GetFFNTileParam(moeConfig);
        TileShape::Current().SetDistTile(
            {batchSize, 1, 0},
            {hiddenSize, 1, 0},
            {ffnTileCnt, ffnTileNum, ffnTailNum});
        auto shmemFlagSched = View(shmemFlag, {1, moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol}, {thisRank, 0, 0, 0});
        auto recvTokenCntOut = DispatchFFNSched(group, flagDummy, shmemFlagSched, moeConfig, ffnTileCnt);
        auto shmemDataBatching = View(shmemData, {1, moeConfig.rankNum, moeConfig.expertNumPerRank, shmemDataLength}, {thisRank, 0 ,0 ,0});
        auto expandXPtr = DispatchFFNBatching(group, tokenTensor, recvTokenCntOut, shmemDataBatching,
            localShmemFlag, expandX.GetShape(0), ffnTileNum + ffnTailNum, moeConfig);
        auto combineInfoPtr = DispatchFFNCombineInfo(group, tokenTensor, recvTokenCntOut, shmemDataBatching,
            localShmemFlag, expandX.GetShape(0), ffnTileNum + ffnTailNum, moeConfig);
        TileShape::Current().SetDistTileRank({moeConfig.expertNumPerRank / 10, 10, 0});
        auto shmemFlagValidCnt = View(shmemFlag, {1, moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol}, {thisRank, 0, 0, 0});
        auto validCntPtr = DispatchFFNValidCnt(recvTokenCntOut, shmemFlagValidCnt, moeConfig);
        expandX = expandXPtr;
        validCnt = validCntPtr;
        combineInfo = combineInfoPtr;
    }
}

void MoeDispatchValidateV2(const Tensor& x, const Tensor& expertIds, const char *group, 
    uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& expandX,
    Tensor& expertTokenNums, Tensor& assistInfoForCombine, Tensor& recvCounts)
{
    std::string assertResult;
    ASSERT(group != nullptr) << "MoeDispatch constraint violated: group name can't be nullptr.";
    ASSERT(group[0] != '\0') << "MoeDispatch constraint violated: group name must be valid, but got '\0'";
    ASSERT(strnlen(group, 128) < 128) << "MoeDispatch constraint violated: group name max size must be 128, but got " << strnlen(group, 128);
    ASSERT(epWorldSize > 0) << "MoeDispatch constraint violated: epWorldSize must be > 0, but got " << epWorldSize;
    ASSERT(moeExpertNum == 160) << "MoeDispatch constraint violated: moeExpertNum must 160, but got " << moeExpertNum;
    ASSERT(sharedExpertNum == 0) << "MoeDispatch constraint violated: sharedExpertNum must 0, but got " << sharedExpertNum;
    ASSERT(sharedExpertRankNum == 0) << "MoeDispatch constraint violated: sharedExpertRankNum must 0, but got " << sharedExpertRankNum;
    int32_t routedExpertNum =  moeExpertNum - sharedExpertNum;
    int32_t expertNumPerRank = routedExpertNum / epWorldSize;
    ASSERT(checkValidInput(x, 2, DataType::DT_BF16, 8, 5120, assertResult)) << assertResult; // 当前仅支持shape:8,5120
    ASSERT(checkValidInput(expertIds, 2, DataType::DT_INT32, 8, 8, assertResult)) << assertResult; // 当前仅支持shape:8,8
    ASSERT(checkValidInput(expertTokenNums, 1, DataType::DT_INT32, expertNumPerRank, 1, assertResult)) << assertResult;
    ASSERT(checkValidInput(recvCounts, 1, DataType::DT_INT32, 1, 0, assertResult)) << assertResult;
    int batchSize = x.GetShape(0);
    int topK = expertIds.GetShape(1);
    int32_t expandXRow = std::min(static_cast<int32_t>(batchSize) *
        static_cast<int32_t>(topK) * static_cast<int32_t>(epWorldSize), static_cast<int32_t>(batchSize) * routedExpertNum);
    ASSERT(checkValidInput(expandX, 2, DataType::DT_BF16, expandXRow, 5120, assertResult)) << assertResult; // 当前仅支持hiddenSize:5120
    ASSERT(checkValidInput(assistInfoForCombine, 2, DataType::DT_INT32, expandXRow, 64, assertResult)) << assertResult; // comBineInfo固定hiddenSize:64
    uint64_t shmemSize = moeExpertNum * x.GetShape(0) * x.GetShape(1) * BytesOf(x.GetDataType()) +
        moeExpertNum * x.GetShape(0) * assistInfoForCombine.GetShape(1) * BytesOf(assistInfoForCombine.GetDataType()) +
        AlignUp(routedExpertNum, 256) * 8 * BytesOf(DataType::DT_INT32) + moeExpertNum * 128 * BytesOf(DataType::DT_INT32) + 128 * BytesOf(DataType::DT_INT32);
    const uint64_t winSize = 1024 * 1024 * 200;
    ASSERT(shmemSize < winSize) << "Exceeds winSize limit. Masxmum allowed: " << winSize << ", got: " << shmemSize;
}

Tensor Nop(const std::vector<Tensor>& inTensors)
{
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto out = std::make_shared<LogicalTensor>(function, DT_INT32, Shape{1, 1});
    LogicalTensors iOperands;
    for (const Tensor& inTensor : inTensors) {
        iOperands.emplace_back(inTensor.GetStorage());
    }
    function.AddOperation(Opcode::OP_NOP, iOperands, {out});
    return out;
}

Tensor DispatchCalcOccurrences(Tensor& expertIds, SymbolicScalar expertId, int32_t calcIndex)
{
    Tensor expertIdsDup = Full(expertId, DT_INT32, {1, expertIds.GetShape(1)});
    Tensor subResult = Sub(expertIdsDup, expertIds);
    Tensor subResultFp32 = Cast(subResult, DT_FP32, CAST_TRUNC);
    Tensor absSubResult = Abs(subResultFp32);
    Tensor subResultInt32 = Cast(absSubResult, DT_INT32, CAST_TRUNC);
    Tensor countOfEquals = Clip(subResultInt32, Element(DT_INT32, 0), Element(DT_INT32, 1));
    Tensor cumSumOffset = CumSum(countOfEquals, 1);
    Tensor cumSumOffsetInt32 = Cast(cumSumOffset, DT_INT32, CAST_TRUNC);
    Tensor expertOffsetResult = ScalarSubS(cumSumOffsetInt32, Element(DT_INT32, calcIndex));
    Tensor expertOffsetResultFp32 = Cast(expertOffsetResult, DT_FP32, CAST_TRUNC);
    Tensor expertOffsetAbsFp32 = Abs(expertOffsetResultFp32);
    Tensor expertOffset = Cast(expertOffsetAbsFp32, DT_INT32, CAST_TRUNC);
    return expertOffset;
}

void MoeDistributedDispatchV2(const Tensor& x, const Tensor& expertIds, const char* group,
    uint32_t epWorldSize, uint32_t moeExpertNum, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum, Tensor& expandX,
    Tensor& assistInfoForCombine, Tensor& expertTokenNums, Tensor& recvCounts)
{
    MoeDispatchValidateV2(x, expertIds, group, epWorldSize, moeExpertNum, sharedExpertNum, 
        sharedExpertRankNum, expandX, expertTokenNums, assistInfoForCombine, recvCounts);
    int32_t routedExpertNum = moeExpertNum - sharedExpertNum;
    ASSERT(epWorldSize > 0) << "MoeDispatch constraint violated: epWorldSize must be > 0, but got " << epWorldSize;
    int32_t expertNumPerRank = routedExpertNum / epWorldSize;
    int32_t batchSize = x.GetShape(0);
    int32_t hiddenSize = x.GetShape(1);
    int32_t topK = expertIds.GetShape(1);
    ASSERT(topK > 0) << "MoeDispatch constraint violated: topK must be > 0, but got " << topK;
    ASSERT(expertNumPerRank > 0) << "MoeDispatch constraint violated: expertNumPerRank must be > 0, but got " << expertNumPerRank;
    Tensor shmemData;
    Tensor shmemInfo;
    Tensor shmemCount;
    Tensor shmemCountSignal;
    Tensor shmemDataSignal;
    int32_t infoSize = assistInfoForCombine.GetShape(1);
    int32_t countSize = 8;
    int32_t signalCol = 128;
    int32_t cumSumRowShape = AlignUp(routedExpertNum, 256);
    SymbolicScalar thisRank = GetHcclRankId(group);

    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void) index;
        Shape shmemDataShape = {expertNumPerRank * epWorldSize, batchSize, hiddenSize};
        Shape shmemInfoShape = {expertNumPerRank * epWorldSize, batchSize, infoSize};
        Shape shmemCountShape = {1, cumSumRowShape, countSize};
        Shape shmemCountSignalShape = {1, moeExpertNum, 1, signalCol};
        Shape shmemDataSignalgShape = {1, 1, 1, signalCol};
        CreateShmemData(group, epWorldSize, x.GetDataType(), shmemDataShape, shmemData, 0);
        CreateShmemData(group, epWorldSize, DT_INT32, shmemInfoShape, shmemInfo, 0);
        CreateShmemData(group, epWorldSize, DT_INT32, shmemCountShape, shmemCount, 0);
        CreateShmemData(group, epWorldSize, DT_INT32, shmemCountSignalShape, shmemCountSignal, 0);
        CreateShmemData(group, epWorldSize, DT_INT32, shmemDataSignalgShape, shmemDataSignal, 0);
    }

    Tensor expertIdsVec = Reshape(expertIds, {1, batchSize * topK});
    Tensor offsetTable(DataType::DT_INT32, {batchSize, topK}, "offsetTable");
    LOOP("MoeDistributedDispatchPrepare", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
        (void) i;
        for (int index = 0; index < batchSize * topK; ++index) {
            int32_t rowIndex = index / topK;
            int32_t colIndex = index % topK;
            TileShape::Current().SetVecTile({1, infoSize});
            SymbolicScalar remoteExpertId = GetTensorData(expertIds, {rowIndex, colIndex});
            Tensor tokenOffsetResult = DispatchCalcOccurrences(expertIdsVec, remoteExpertId, index);
            SymbolicScalar tokenOffset = GetTensorData(tokenOffsetResult, {0, index - 1});
            SetTensorData(tokenOffset, {rowIndex, colIndex}, offsetTable);
        }
    }

    LOOP("MoeDistributedDispatchSendData", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
        (void) i;
        for (int32_t index = 0; index < topK * batchSize; ++index) {
            Tensor moeInfo(DataType::DT_INT32, {1, infoSize}, "moeInfo");
            int32_t rowIndex = index / topK;
            int32_t colIndex = index % topK;
            Tensor tensorTile = View(x, {1, hiddenSize}, {rowIndex, 0});
            TileShape::Current().SetVecTile({1, infoSize});
            SetTensorData(thisRank, {0, assistInfoForCombine.GetShape(1) - 3}, moeInfo);
            SetTensorData(rowIndex, {0, assistInfoForCombine.GetShape(1) - 2}, moeInfo);
            SetTensorData(colIndex, {0, assistInfoForCombine.GetShape(1) - 1}, moeInfo);
            SymbolicScalar remoteExpertId = GetTensorData(expertIds, {rowIndex, colIndex});
            SymbolicScalar remoteExpertOffset = remoteExpertId % expertNumPerRank;
            SymbolicScalar remoteRankId = remoteExpertId / expertNumPerRank;
            SymbolicScalar tokenOffset = GetTensorData(offsetTable, {rowIndex, colIndex});
            Tensor shmemDataTile = View(shmemData, {1, 1, 1, hiddenSize}, std::vector<SymbolicScalar>{remoteRankId, remoteExpertOffset * epWorldSize + thisRank, tokenOffset, 0});
            TileShape::Current().SetVecTile({1, hiddenSize});
            Tensor shmemDataPutOut = ShmemPut(offsetTable, tensorTile, shmemDataTile);
            Tensor shmemInfoTile = View(shmemInfo, {1, 1, 1, infoSize}, std::vector<SymbolicScalar>{remoteRankId, remoteExpertOffset * epWorldSize + thisRank, tokenOffset, 0});
            TileShape::Current().SetVecTile({1, infoSize});
            Tensor shmemInfoPutOut = ShmemPut(offsetTable, moeInfo, shmemInfoTile);
            Tensor sendOut = Nop({shmemDataPutOut, shmemInfoPutOut});
            TileShape::Current().SetVecTile({1, signalCol});
            Tensor shmemDataSignalTile = View(shmemDataSignal, {epWorldSize, 1, 1, 1, signalCol}, {0, 0, 0, 0, 0});
            ShmemSignal(sendOut, shmemDataSignalTile, AtomicType::ADD);
        }
    }

    Tensor shmemCountOut(DT_INT32, {1, 1}, "shmemCountOut");
    LOOP("MoeDistributedDispatchSendCount", FunctionType::DYNAMIC_LOOP, expertId, LoopRange(moeExpertNum)) {
        Tensor expertOffset = DispatchCalcOccurrences(expertIdsVec, expertId, batchSize * topK);
        TileShape::Current().SetVecTile({1, 1});
        SymbolicScalar remoteRankId = expertId / expertNumPerRank;
        SymbolicScalar remoteExpertOffset = expertId % expertNumPerRank;
        Tensor shmemCountTile = View(shmemCount, {1, 1, 1, 1}, {remoteRankId, 0, remoteExpertOffset * epWorldSize + thisRank + 1, 0});
        Tensor totalOffsetTile = View(expertOffset, {1, 1}, {0, batchSize * topK - 1});
        Tensor shmemPutOut = ShmemPut(totalOffsetTile, totalOffsetTile, shmemCountTile);
        TileShape::Current().SetVecTile({1, signalCol});
        Tensor shmemCountSignalTile = View(shmemCountSignal, {1, 1, 1, 1, signalCol}, {remoteRankId, 0, 0, 0, 0});
        shmemCountOut = ShmemSignal(shmemPutOut, shmemCountSignalTile, AtomicType::ADD);
    }

    Tensor cumSumResult(DT_INT32, {cumSumRowShape, countSize}, "cumSumResult");
    Tensor localExpertRecvCount(DT_INT32, {cumSumRowShape, countSize}, "localExpertRecvCount");
    LOOP("MoeDistributedDispatchCumSum", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
        (void) i;
        TileShape::Current().SetVecTile({1, signalCol});
        Tensor shmemDataSignalLocalTile = View(shmemDataSignal, {1, 1, 1, 1, signalCol}, {thisRank, 0, 0, 0, 0});
        Tensor waitUntilOut1 = WaitUntil(cumSumResult, shmemDataSignalLocalTile, batchSize * topK * epWorldSize, true);
        TileShape::Current().SetVecTile({1, signalCol});
        Tensor shmemCountSignalLocalTile = View(shmemCountSignal, {1, 1, 1, 1, signalCol}, {thisRank, 0, 0, 0, 0});
        Tensor waitUntilOut = WaitUntil(cumSumResult, shmemCountSignalLocalTile, moeExpertNum, true);
        Tensor waitOut = Nop({waitUntilOut1, waitUntilOut});

        TileShape::Current().SetVecTile({1, countSize});
        Tensor shmemReceiveCountTile = View(shmemCount, {1, 1, cumSumRowShape, countSize}, {thisRank, 0, 0, 0});
        localExpertRecvCount = ShmemGet(waitOut, shmemReceiveCountTile);
        TileShape::Current().SetVecTile({cumSumRowShape, countSize});
        Tensor shmemCountTile = View(shmemCount, {1, 1, cumSumRowShape, countSize}, {thisRank, 0, 0, 0});
        Tensor shmemGetOut = ShmemGet(waitOut, shmemCountTile);
        Tensor cumSumCurrent = CumSum(shmemGetOut, 0);
        cumSumResult = Cast(cumSumCurrent, DT_INT32, CAST_TRUNC);

        SymbolicScalar recvCountResult = GetTensorData(cumSumResult, {expertNumPerRank * epWorldSize, 0});
        SetTensorData(recvCountResult, {0}, recvCounts);

        for (int32_t expertId = 0; expertId < expertNumPerRank; ++expertId) {
            Tensor expertValidCnt = View(shmemGetOut, {epWorldSize, countSize}, {expertId * epWorldSize + 1, 0});
            Tensor expertValidCumSum = CumSum(expertValidCnt, 0);
            Tensor expertCumSumResult = Cast(expertValidCumSum, DT_INT32, CAST_TRUNC);
            SymbolicScalar recvValidResult = GetTensorData(expertCumSumResult, {epWorldSize - 1, 0});
            SetTensorData(recvValidResult, {expertId}, expertTokenNums);
        }
    }

    LOOP("MoeDistributedDispatchReceive", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
        (void) i;
        for (uint32_t index = 0; index < expertNumPerRank * epWorldSize; ++index) {
            SymbolicScalar curCount = GetTensorData(localExpertRecvCount, {index + 1, 0});
            SymbolicScalar offset = GetTensorData(cumSumResult, {index, 0});
            Tensor curShmemDataTile = View(shmemData, {1, 1, batchSize, hiddenSize}, std::vector<SymbolicScalar>{1, 1, curCount, hiddenSize}, {thisRank, index, 0, 0});
            TileShape::Current().SetVecTile({batchSize, hiddenSize});
            Tensor localDataRecvCount = ShmemGetGm2Ub(cumSumResult, curShmemDataTile);
            Assemble(localDataRecvCount, std::vector<SymbolicScalar>{offset, 0}, expandX);
            Tensor curShmemInfoTile = View(shmemInfo, {1, 1, batchSize, infoSize}, std::vector<SymbolicScalar>{1, 1, curCount, infoSize}, {thisRank, index, 0, 0});
            TileShape::Current().SetVecTile({batchSize, infoSize});
            Tensor localInfoRecvCount = ShmemGetGm2Ub(cumSumResult, curShmemInfoTile);
            Assemble(localInfoRecvCount, std::vector<SymbolicScalar>{offset, 0}, assistInfoForCombine);
        }
    }
}
}
}