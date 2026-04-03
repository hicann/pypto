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
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "distributed_expand.h"
#include "tilefwk/comm_group_recorder.h"
#include "interface/utils/distributed_error.h"

namespace npu::tile_fwk {
namespace Distributed {
constexpr int32_t DIST_HEAD_SHAPE = 0;
constexpr int32_t DIST_HEAD_COUNT = 1;
constexpr int32_t DIST_TAIL_SHAPE = 2;
constexpr int32_t DIST_INDEX_ZERO = 0;
constexpr int32_t DIST_INDEX_ONE = 1;
constexpr int32_t DIST_INDEX_TWO = 2;
constexpr uint16_t COPY_BLOCK_BYTE_SIZE = 32;
constexpr uint16_t SAME_ADDR_BYTE_SIZE = 512;
constexpr uint64_t SHMEM_SIZE_ALIGN = 512;
constexpr int32_t ROUTED_EXPET_NUM = 160;
constexpr int32_t FFN_TILE_SIZE = 8;
constexpr int32_t AIV_NUM = 4;
constexpr int32_t RECEIVE_CNT_OUT_ROW = 1024;
constexpr int32_t RECEIVE_CNT_OUT_COL = 512;
constexpr int32_t SHMEM_SIGNAL_STRIDE = 8;
enum class TileIndex : size_t { HEAD_SHAPE, HEAD_NUM, TAIL_SHAPE };

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

inline std::string OpTypeToString(OpType type)
{
    switch (type) {
        case OpType::EQ:
            return "OpType::EQ";
        case OpType::NE:
            return "OpType::NE";
        case OpType::LT:
            return "OpType::LT";
        case OpType::LE:
            return "OpType::LE";
        case OpType::GT:
            return "OpType::GT";
        case OpType::GE:
            return "OpType::GE";
        default:
            return "";
    }
}

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

template <typename T>
typename std::enable_if<!is_iterable_v<T>, std::string>::type ToString(T value)
{
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else if constexpr (std::is_convertible_v<T, std::string>) {
        return std::string(value);
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(value);
    } else if constexpr (std::is_same_v<T, AtomicType>) {
        return AtomicTypeToString(value);
    } else if constexpr (std::is_same_v<T, DataType>) {
        return DataType2String(value);
    } else if constexpr (std::is_same_v<T, Opcode>) {
        return OpcodeManager::Inst().GetOpcodeStr(value);
    } else if constexpr (std::is_same_v<T, OpType>) {
        return OpTypeToString(value);
    } else {
        return "";
    }
}

template <typename Container>
typename std::enable_if<is_iterable_v<Container>, std::string>::type ToString(const Container& c)
{
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& item : c) {
        if (!first) {
            oss << ", ";
        }
        oss << ToString(item);
        first = false;
    }
    oss << "]";
    return oss.str();
}

struct ShmemPutAttr {
    Shape copyBufferShape;
    AtomicType atomicType = AtomicType::SET;
    SymbolicScalar ownerRank;
};

struct ShmemGetAttr {
    Shape copyBufferShape;
    AtomicType atomicType = AtomicType::SET;
    SymbolicScalar ownerRank;
};

struct ShmemSignalAttr {
    int64_t signalValue = 1;
    int32_t signalStride = SHMEM_SIGNAL_STRIDE;
    int64_t tileRowShape = 0;
    int64_t tileColShape = 0;
    AtomicType atomicType = AtomicType::SET;
    bool notifyAll{false};
    int64_t worldSize{0};
    SymbolicScalar ownerRank;
};

struct ShmemWaitUntilAttr {
    int32_t expectedSum = 0;
    int32_t signalStride = SHMEM_SIGNAL_STRIDE;
    bool resetSignal = false;
    int64_t tileRowShape = 0;
    int64_t tileColShape = 0;
    SymbolicScalar ownerRank;
};

struct ShmemSetAttr {
    bool isSetData{true};
    Shape setBufferShape;
    SymbolicScalar ownerRank;
};

struct MoeDispatchAttr {
    std::string extraTemplateParam{};
    int64_t topK = 0;
    SymbolicScalar ownerRank;
};

struct MoeCombineAttr {
    int64_t setType = 0;
    int64_t topK = 0;
    int64_t paddedColShape{0};
    int64_t rowOffset{-1};
    int64_t rowShape{-1};
    SymbolicScalar ownerRank;
};

inline int GetTotalTileNum(const std::array<int, MAX_DIST_DIM_SIZE>& tile)
{
    return tile[static_cast<size_t>(TileIndex::HEAD_NUM)] +
           static_cast<int>(tile[static_cast<size_t>(TileIndex::TAIL_SHAPE)] != 0);
}

inline bool checkValidInput(
    const Tensor& input, uint64_t dim, DataType dType, int32_t row, int32_t col, std::string& assertResult)
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
        assertResult =
            "Distributed constraint violated: " + input.GetName() + " dim must be " + std::to_string(dim) + ".";
        return false;
    }
    if (input.GetDataType() != dType) {
        assertResult = "Distributed constraint violated: " + input.GetName() + " dataType is not valid.";
        return false;
    }
    if (input.GetShape(0) != row) {
        assertResult =
            "Distributed constraint violated: " + input.GetName() + " row must be " + std::to_string(row) + ".";
        return false;
    }
    if (input.Dim() != 1 && input.GetShape(1) != col) {
        assertResult =
            "Distributed constraint violated: " + input.GetName() + " col must be " + std::to_string(col) + ".";
        return false;
    }
    return true;
}

inline bool checkValidConfig(const MoeConfig& moeConfig, std::string& assertResult)
{
    int32_t rankNum = moeConfig.rankNum;
    int32_t routedExpertNum = moeConfig.routedExpertNum;
    int32_t expertNumPerRank = moeConfig.expertNumPerRank;
    if (rankNum != 4 && rankNum != 8) { // rankNum仅支持4和8
        assertResult = "Distributed constraint violated: moeConfig rankSize must be 4 or 8.";
        return false;
    }
    if (routedExpertNum != ROUTED_EXPET_NUM) {
        assertResult = "Distributed constraint violated: moeConfig routedExpertNum must be " +
                       std::to_string(ROUTED_EXPET_NUM) + ".";
        return false;
    }
    if (expertNumPerRank != routedExpertNum / rankNum) {
        assertResult = "Distributed constraint violated: moeConfig expertNumPerRank must be " +
                       std::to_string(routedExpertNum / rankNum) + ".";
        return false;
    }
    return true;
}

inline int64_t GetTotalTileNum(const VecTile& tileShape, const Shape& dataShape)
{
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, tileShape.size() == 2)
        << "Invalid dimensional: "
        << " tileShape dim must be 2, but got dimensional=" << tileShape.size();
    ASSERT(DistributedErrorCode::INVALID_TENSOR_DIM, dataShape.size() >= 2)
        << "Invalid dimensional: "
        << " dataShape dim must >= 2, but got dimensional=" << dataShape.size();
    auto totalRowShape = dataShape[dataShape.size() - 2];
    auto totalColShape = dataShape[dataShape.size() - 1];
    auto tileRowShape = tileShape[0];
    auto tileColShape = tileShape[1];
    auto tileRowNum = totalRowShape / tileRowShape + (totalRowShape % tileRowShape == 0 ? 0 : 1);
    auto tileColNum = totalColShape / tileColShape + (totalColShape % tileColShape == 0 ? 0 : 1);
    return tileRowNum * tileColNum;
}
} // namespace Distributed
} // namespace npu::tile_fwk

#endif
