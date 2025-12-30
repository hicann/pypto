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
 * \file distributed_common.h
 * \brief
 */

#ifndef DISTRIBUTED_COMMON_H
#define DISTRIBUTED_COMMON_H

#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include "interface/inner/config.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "distributed_expand.h"
#include "tilefwk/comm_group_recorder.h"

namespace npu::tile_fwk {
namespace Distributed {
constexpr int32_t FLAG_TENSOR_SIZE = 64;
constexpr int32_t DIST_HEAD_SHAPE = 0;
constexpr int32_t DIST_HEAD_COUNT = 1;
constexpr int32_t DIST_TAIL_SHAPE = 2;
constexpr int32_t DIST_INDEX_ZERO = 0;
constexpr int32_t DIST_INDEX_ONE = 1;
constexpr int32_t DIST_INDEX_TWO = 2;
constexpr int32_t DIST_INDEX_THREE = 3;
constexpr uint16_t COPY_BLOCK_BYTE_SIZE = 32;
constexpr uint16_t SAME_ADDR_BYTE_SIZE = 512;
constexpr int32_t ROUTED_EXPET_NUM = 160;
constexpr int32_t AIV_MAX_NUM = 8;
constexpr int32_t AIV_NUM = 4;
constexpr int32_t RECEIVE_CNT_OUT_ROW = 1024;
constexpr int32_t RECEIVE_CNT_OUT_COL = 512;
enum class TileIndex : size_t {
    HEAD_SHAPE,
    HEAD_NUM,
    TAIL_SHAPE
};

enum class AtomicType {
    SET,
    ADD
};

enum class AllReduceType {
    ONE_SHOT,
    TWO_SHOT,
};

inline std::string AtomicTypeToString(AtomicType type)
{
    switch (type) {
        case AtomicType::SET:
            return "TileOp::Distributed::AtomicType::SET";
        case AtomicType::ADD:
            return "TileOp::Distributed::AtomicType::ADD";
        default:
            return "";
    }
}

struct DistOpAttr {
public:
    AtomicType atomicType = AtomicType::SET;
    int64_t signalValue;
    std::vector<int64_t> aicpuOpParams;
    bool fp32Mode;
    int64_t topK;
    Shape copyBufferShape;
    Shape setBufferShape;
    std::string extraTemplateParam{};
    int64_t paddedColShape;
};

inline int GetTotalTileNum(const std::array<int, MAX_DIST_DIM_SIZE> &tile)
{
    return tile[static_cast<size_t>(TileIndex::HEAD_NUM)] +
        static_cast<int>(tile[static_cast<size_t>(TileIndex::TAIL_SHAPE)] != 0);
}

inline int GetTotalSize(const std::array<int, MAX_DIST_DIM_SIZE> &tile)
{
    return tile[static_cast<size_t>(TileIndex::HEAD_SHAPE)] * tile[static_cast<size_t>(TileIndex::HEAD_NUM)] +
        tile[static_cast<size_t>(TileIndex::TAIL_SHAPE)];
}

inline bool NotNegative(const std::array<int, MAX_DIST_DIM_SIZE> &tile)
{
    return (tile[static_cast<size_t>(TileIndex::HEAD_SHAPE)] >= 0) &&
        (tile[static_cast<size_t>(TileIndex::HEAD_NUM)] >= 0) &&
        (tile[static_cast<size_t>(TileIndex::TAIL_SHAPE)] >= 0);
}

struct DispatchTilingInfo {
    int tileIndex{0};
    int groupIndex{0};
    int shape{0};
    int offset{0};
    int totalTileNum{0};
    int magic{0x5a5a5a5a};

    std::vector<int> SerializeTo() const
    {
        return std::vector<int>{
            tileIndex, groupIndex,
            shape, offset,
            totalTileNum, magic
        };
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "tileIndex=" << tileIndex
            << ", groupIndex=" << groupIndex
            << ", shape=" << shape
            << ", offset=" << offset
            << ", totalTileNum=" << totalTileNum;
        return ss.str();
    }
};

struct TilingInfo {
    int tileIndex{0};
    int groupIndex{0};
    int rowPerRank{0};
    int colPerRank{0};
    int rankShape{0};
    int rankOffset{0};
    int rowShape{0};
    int rowOffset{0};
    int colShape{0};
    int colOffset{0};
    int totalTileNum{0};
    int shareRankCnt{0};
    int magic{0x5a5a5a5a};

    std::vector<int> SerializeTo() const
    {
        return std::vector<int>{
            tileIndex, groupIndex,
            rowPerRank, colPerRank,
            rankShape, rankOffset,
            rowShape, rowOffset,
            colShape, colOffset,
            totalTileNum, shareRankCnt, magic
        };
    }
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "tileIndex=" << tileIndex
            << ", rowShape=" << rowShape
            << ", colShape=" << colShape
            << ", rowOffset=" << rowOffset
            << ", colOffset=" << colOffset
            << ", rankShape=" << rankShape
            << ", rankOffset=" << rankOffset;
        return ss.str();
    }
};

template <typename T = TilingInfo>
struct OpArgs {
    std::string opName;
    LogicalTensors iOperands;
    LogicalTensors oOperands;
    std::shared_ptr<LogicalTensor> tilingTensor;
    std::string tilingSymbol;
    std::optional<T> tilingInfo;
    std::optional<std::vector<int64_t>> attrArray;

    void PrintLog() const
    {
        ALOG_INFO_F("Distributed Op name=[%s]", opName.c_str());
        for (uint32_t i = 0; i < iOperands.size(); i++) {
            ALOG_INFO_F("iOperands[%u] symbol=[%s], magic=[%d %d], shape=[%d %d], offset=[%d %d]",
                i,
                iOperands[i]->Symbol().c_str(),
                iOperands[i]->GetMagic(), iOperands[i]->GetRawMagic(),
                iOperands[i]->shape[0], iOperands[i]->shape[1],
                iOperands[i]->offset[0], iOperands[i]->offset[1]);
        }
        for (uint32_t i = 0; i < oOperands.size(); i++) {
            ALOG_INFO_F("oOperands[%u] symbol=[%s], magic=[%d %d], shape=[%d %d], offset=[%d %d]",
                i,
                oOperands[i]->Symbol().c_str(),
                oOperands[i]->GetMagic(), oOperands[i]->GetRawMagic(),
                oOperands[i]->shape[0], oOperands[i]->shape[1],
                oOperands[i]->offset[0], oOperands[i]->offset[1]);
        }
        if (tilingInfo.has_value()) {
            auto &tilingValue = tilingInfo.value();
            ALOG_INFO_F("tilingInfo: %s", tilingValue.ToString().c_str());
        }
    }
};

struct TensorTileInfo {
    std::array<int, MAX_DIST_DIM_SIZE> row;
    std::array<int, MAX_DIST_DIM_SIZE> col;
};

struct CommGroupInfo {
    int groupIndex{0};
    std::optional<std::array<int, MAX_DIST_DIM_SIZE>> rank{std::nullopt};
    std::optional<int> rankSize{std::nullopt};
    std::optional<int> rankId{std::nullopt};
    CommGroupInfo(const char *group, const TileShape &tileShape)
    {
        groupIndex = static_cast<int32_t>(CommGroupRecorder::GetInstance().Input(std::string(group)));
        if (tileShape.GetDistRankId() >= 0) {
            rankId = std::make_optional(tileShape.GetDistRankId());
        }
        auto& tempRank = tileShape.GetDistTileRank();
        if ((NotNegative(tempRank)) && GetTotalSize(tempRank) > 0) {
            rank = std::make_optional(tempRank);
            rankSize = std::make_optional(GetTotalSize(tempRank));
        }
    }
    CommGroupInfo() = default;

    bool CheckAndUpdate(std::optional<int> size)
    {
        if (size.has_value() && rankSize.has_value()) {
            return size.value() == rankSize.value();
        }
        if (size.has_value() && (!rankSize.has_value())) {
            rankSize = size;
            rank = std::make_optional(std::array<int, MAX_DIST_DIM_SIZE>{1, rankSize.value(), 0});
            return true;
        }
        return rankSize.has_value() && rankId.has_value();
    }
};

class DistTensorTilingInfo {
public:
    std::vector<std::array<int, MAX_DIST_DIM_SIZE>> tileInfo;
    DistTensorTilingInfo() = default;
    explicit DistTensorTilingInfo(const TileShape &tileShape, size_t dim)
    {
        if (dim > 0) {
            tileInfo.push_back(tileShape.GetDistTileRow());
        }
        if (dim > 1) {
            tileInfo.push_back(tileShape.GetDistTileCol());
        }
    }
    bool Check(const std::vector<std::optional<int>> &checker) const
    {
        if (checker.size() != tileInfo.size()) {
            return false;
        }
        for (size_t i = 0; i < checker.size(); ++i) {
            if (!Check(i, checker[i])) {
                return false;
            }
        }
        return true;
    }

    bool Check(size_t index, std::optional<int> total) const
    {
        if (!NotNegative(tileInfo[index])) {
            return false;
        }
        auto sum = GetTotalSize(tileInfo[index]);
        if (sum == 0) {
            return false;
        }
        if (total.has_value() && sum != total.value()) {
            return false;
        }
        return true;
    }
    auto& operator[](size_t index) const
    {
        return tileInfo[index];
    }
};

struct TileArgs {
    Function &function;
    std::shared_ptr<LogicalTensor> in;
    std::shared_ptr<LogicalTensor> tilingTensor;
    std::shared_ptr<LogicalTensor> out;
    TilingInfo &tilingInfo;
    const TensorTileInfo &tileInfo;
    const CommGroupInfo &groupInfo;
    const std::string &tilingSymbol;
};
using DealTileFunc = std::function<void(TileArgs &)>;

void CheckAndGetGroupInfo(const int groupIndex, const TileShape &tileShape, CommGroupInfo &groupInfo);

void CheckAndGetTileInfo(int rowTotal, int colTotal, const TileShape &tileShape, TensorTileInfo &tileInfo);

template <typename T = TilingInfo>
Operation &AddOperation(Function& function, OpArgs<T> &opArgs)
{
    if (opArgs.tilingInfo.has_value() && (opArgs.tilingTensor != nullptr)) {
        auto tilingData = opArgs.tilingInfo.value().SerializeTo();
        int offset = function.GetDistTilingManager()->Save(opArgs.tilingSymbol, tilingData);
        ASSERT(offset >= 0);
        const std::vector<int64_t> newShape = {1, static_cast<int>(tilingData.size())};
        const std::vector<int64_t> newOffset = {0, offset};
        auto tiling = opArgs.tilingTensor->View(function, newShape, newOffset);
        opArgs.iOperands.push_back(tiling);
    }

    auto &oper = function.AddOperation(opArgs.opName, opArgs.iOperands, opArgs.oOperands);

    if (opArgs.attrArray.has_value()) {
        oper.SetAttribute(OP_ATTR_PREFIX + "distributed", opArgs.attrArray.value());
    }

    if (IsEmptyOut(oper.GetOpcode())) {
        oper.SetAttr(OpAttributeKey::dontTouch, true);
    }

    opArgs.PrintLog();
    return oper;
}

template <typename T>
void TileColProcess(const std::function<void(T &)> &dealFunc, T &args)
{
    const std::array<int32_t, MAX_DIST_DIM_SIZE> &colTileInfo = args.tileInfo.col;

    const int32_t tileColShape = colTileInfo[0];
    const int32_t tileColCnt = colTileInfo[1];
    const int32_t tailColShape = colTileInfo[2];

    // 列头块
    for (int32_t colTileIndex = 0; colTileIndex < tileColCnt; colTileIndex++) {
        args.tilingInfo.colShape = tileColShape;
        args.tilingInfo.colOffset = colTileIndex * tileColShape;
        dealFunc(args);
        args.tilingInfo.tileIndex++;
    }
    // 列尾块
    if (tailColShape != 0) {
        args.tilingInfo.colShape = tailColShape;
        args.tilingInfo.colOffset = tileColCnt * tileColShape;
        dealFunc(args);
        args.tilingInfo.tileIndex++;
    }
    return;
}

template <typename T>
void TileColAndRowProcess(const std::function<void(T &)> &dealFunc, T &args)
{
    // 每个调用 TileColAndRowProcess 的回调都是单独的 OP，tileIndex 都需要单独置 0
    args.tilingInfo.tileIndex = 0;

    const std::array<int, MAX_DIST_DIM_SIZE>& rowTileInfo = args.tileInfo.row;

    const int32_t tileRowShape = rowTileInfo[0];
    const int32_t tileRowCnt = rowTileInfo[1];
    const int32_t tailRowShape = rowTileInfo[2];
    // 行头块处理
    for (int32_t rowTileIndex = 0; rowTileIndex < tileRowCnt; rowTileIndex++) {
        args.tilingInfo.rowShape = tileRowShape;
        args.tilingInfo.rowOffset = rowTileIndex * tileRowShape;
        TileColProcess(dealFunc, args);
    }
    // 行尾块处理
    if (tailRowShape != 0) {
        args.tilingInfo.rowShape = tailRowShape;
        args.tilingInfo.rowOffset = tileRowCnt * tileRowShape;
        TileColProcess(dealFunc, args);
    }

    return;
}

inline bool checkValidInput(const Tensor &input, uint64_t dim, DataType dType, int32_t row, int32_t col, std::string &assertResult)
{
    if (input.Format() != TileOpFormat::TILEOP_ND) {
        assertResult = "Distributed constraint violated: " + input.GetName() + " format must be TILEOP_ND.";
        return false;
    }
    if (input.GetName() == "") {
        assertResult = "Distributed constraint violated: input name can't be null.";
        return false;
    }
    if (input.Dim() != dim) {
        assertResult = "Distributed constraint violated: " + input.GetName() + " dim must be " + std::to_string(dim) + ".";
        return false;
    }
    if (input.GetDataType() != dType) {
        assertResult = "Distributed constraint violated: " + input.GetName() + " dataType is not valid.";
        return false;
    }
    if (input.GetShape(0) != row) {
        assertResult = "Distributed constraint violated: " + input.GetName() + " row must be " + std::to_string(row) + ".";
        return false;
    }
    if (input.Dim() != 1 && input.GetShape(1) != col) {
        assertResult = "Distributed constraint violated: " + input.GetName() + " col must be " + std::to_string(col) + ".";
        return false;
    }
    return true;
}

inline bool checkValidConfig(const MoeConfig &moeConfig, std::string &assertResult)
{
    int32_t rankNum = moeConfig.rankNum;
    int32_t routedExpertNum = moeConfig.routedExpertNum;
    int32_t expertNumPerRank = moeConfig.expertNumPerRank;
    if (rankNum != 4 && rankNum != 8) { // rankNum仅支持4和8
        assertResult = "Distributed constraint violated: moeConfig rankSize must be 4 or 8.";
        return false;
    }
    if (routedExpertNum != ROUTED_EXPET_NUM) {
        assertResult = "Distributed constraint violated: moeConfig routedExpertNum must be " + std::to_string(ROUTED_EXPET_NUM) + ".";
        return false;
    }
    if (expertNumPerRank != routedExpertNum / rankNum) {
        assertResult = "Distributed constraint violated: moeConfig expertNumPerRank must be " + std::to_string(routedExpertNum / rankNum) + ".";
        return false;
    }
    return true;
}

int GetTilingTensorSize(const TensorTileInfo &tileInfo, const CommGroupInfo &groupInfo);

} // namespace Distributed
} // namespace npu::tile_fwk

#endif
