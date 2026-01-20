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

Tensor CreateShmem(int32_t rankSize, int32_t expertNumPerRank, int32_t shmemCol, int32_t hcclGroupIndex, 
    DataType dataType, uint32_t memType)
{
    auto &function = *Program::GetInstance().GetCurrentFunction();
    Shape shmemShape;
    if (memType == 0) {
        shmemShape = {rankSize, rankSize, expertNumPerRank, shmemCol};
    } else {
        shmemShape = {rankSize, expertNumPerRank, rankSize, shmemCol};
    }
    auto shmemTensor = std::make_shared<LogicalTensor>(function, dataType, shmemShape);
    auto &op = function.AddOperation(Opcode::OP_BIND_TENSOR, {}, {shmemTensor});
    op.SetAttribute(OpAttributeKey::bindTensor, BindTensor(hcclGroupIndex, 0,
        BytesOf(dataType) * rankSize * expertNumPerRank * shmemCol));
    return shmemTensor;
}

std::tuple<int32_t, int32_t, int32_t> GetFFNTileParam(const MoeConfig &moeConfig)
{
    int32_t tileRankCnt = moeConfig.rankNum > FFN_TILE_SIZE ? FFN_TILE_SIZE : moeConfig.rankNum;
    int32_t tileNum = tileRankCnt == FFN_TILE_SIZE ? moeConfig.rankNum / FFN_TILE_SIZE : 1;
    int32_t tailNum = tileNum == 1 ? 0 : (moeConfig.rankNum % FFN_TILE_SIZE == 0 ? 0 : 1);
    return {tileRankCnt, tileNum, tailNum};
}

void MoeDispatch(const Tensor &tokenTensor, const Tensor &tokenExpertTable, Tensor &expandX,
    Tensor &validCnt, Tensor &combineInfo, const char *group, const MoeConfig &moeConfig)
{
    std::string assertResult;
    ASSERT(checkValidConfig(moeConfig, assertResult)) << assertResult;
    ASSERT(group != nullptr) << "MoeDispatch constraint violated: group name can't be nullptr.";
    ASSERT(group[0] != '\0') << "MoeDispatch constraint violated: group name is not valid.";
    ASSERT(strnlen(group, 128) < 128) << "MoeDispatch constraint violated: group name max size must be 128.";

    ASSERT(checkValidInput(tokenTensor, 2, DataType::DT_BF16, 8, 5120, assertResult)) << assertResult; // 当前仅支持shape:8,5120
    ASSERT(checkValidInput(tokenExpertTable, 2, DataType::DT_INT32, 8, 8, assertResult)) << assertResult; // 当前仅支持shape:8,8
    ASSERT(checkValidInput(validCnt, 1, DataType::DT_INT32, moeConfig.expertNumPerRank, 1, assertResult)) << assertResult;

    int hcclGroupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
    SymbolicScalar thisRank = GetHcclRankId(hcclGroupIndex);
    int batchSize = tokenTensor.GetShape(0);
    int hiddenSize = tokenTensor.GetShape(1);
    int topK = tokenExpertTable.GetShape(1);
    int shmemDataLength = AlignUp(hiddenSize, 512) + 512;
    int32_t expandXRow = std::min(static_cast<int32_t>(batchSize) *
        static_cast<int32_t>(topK) * moeConfig.rankNum, static_cast<int32_t>(batchSize) * moeConfig.routedExpertNum);

    ASSERT(checkValidInput(expandX, 2, DataType::DT_BF16, expandXRow, 5120, assertResult)) << assertResult; // 当前仅支持hiddenSize:5120
    ASSERT(checkValidInput(combineInfo, 2, DataType::DT_INT32, expandXRow, 3, assertResult)) << assertResult; // comBineInfo固定hiddenSize:3

    int flagRow = 1;
    int flagCol = 128;
    Tensor shmemData;
    Tensor shmemFlag;
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void) index;
        int32_t shmemDataCol = shmemDataLength * batchSize;
        shmemData = CreateShmem(moeConfig.rankNum, moeConfig.expertNumPerRank, shmemDataCol, 
            hcclGroupIndex, tokenTensor.GetDataType(), 0);
        shmemFlag = CreateShmem(moeConfig.rankNum, moeConfig.expertNumPerRank, flagCol, 
            hcclGroupIndex, DT_INT32, 1);
    }
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
        auto shmemFlagSched = View(shmemFlag, {1, moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol},
            {thisRank, 0, 0, 0});
        auto recvTokenCntOut = DispatchFFNSched(group, flagDummy, shmemFlagSched, moeConfig, ffnTileCnt);
        auto shmemDataBatching = View(shmemData, {1, moeConfig.rankNum, moeConfig.expertNumPerRank, shmemDataLength},
            {thisRank, 0 ,0 ,0});
        auto expandXPtr = DispatchFFNBatching(group, tokenTensor, recvTokenCntOut, shmemDataBatching,
            localShmemFlag, expandX.GetShape(0), ffnTileNum + ffnTailNum, moeConfig);
        auto combineInfoPtr = DispatchFFNCombineInfo(group, tokenTensor, recvTokenCntOut, shmemDataBatching,
            localShmemFlag, expandX.GetShape(0), ffnTileNum + ffnTailNum, moeConfig);
        TileShape::Current().SetDistTileRank({moeConfig.expertNumPerRank / 10, 10, 0});
        auto shmemFlagValidCnt = View(shmemFlag, {1, moeConfig.expertNumPerRank, moeConfig.rankNum, flagCol},
            {thisRank, 0, 0, 0});
        auto validCntPtr = DispatchFFNValidCnt(recvTokenCntOut, shmemFlagValidCnt, moeConfig);
        expandX = expandXPtr;
        validCnt = validCntPtr;
        combineInfo = combineInfoPtr;
    }
}
}
}