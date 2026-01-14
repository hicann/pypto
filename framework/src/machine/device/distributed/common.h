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
 * \file common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include "tilefwk/aicore_data.h"
#include "machine/utils/dynamic/dev_encode_types.h"

namespace npu::tile_fwk::dynamic {
    class AiCoreManager;
}

namespace npu::tile_fwk::Distributed {
constexpr uint64_t AICPU_TASK_ARRAY_SIZE = 1024;
constexpr uint64_t AICPU_TASK_ARRAY_SIZE_MOD = AICPU_TASK_ARRAY_SIZE - 1;
constexpr uint64_t SRC_SHMEM_SIGNAL_ID = 1;
constexpr uint64_t SRC_RANK_ID = 2;
constexpr uint64_t SHMEM_DIM_ROW = 3;
constexpr uint64_t SHMEM_DIM_COL = 4;

struct TensorInfo {
    uint64_t rawAddr{0};
    uint32_t dim{0};
    uint64_t rawIndex{0};
    int32_t expectedSum{0};
    int32_t signalStride{0};
    bool resetSignal{false};
    std::vector<uint32_t> offset;
    std::vector<uint32_t> shape;
};

struct AicpuParamInfo {
    int32_t outIndex{0};
    int32_t inIndex{0};
    int32_t attrIndex{0};
    int32_t rawShapeIndex{0};
    int32_t tileShapeIndex{0};
    uint32_t rawShapeRow{0};
    uint32_t rawShapeCol{0};
    uint32_t rawRankShape{0};
    uint32_t tileShapeRow{0};
    uint32_t tileShapeCol{0};
};

inline uint64_t GetVirtualAddrBist(uint64_t val, uint64_t start, uint64_t end) {
    return (((val) >> (start)) & ((1UL << ((end) - (start) + 1UL)) - 1UL));
}

inline uint64_t GetVirtualAddrOffset(uint64_t val) {
    constexpr uint64_t offsetStart = 0UL; 
    constexpr uint64_t offsetEnd = 57UL; 
    return GetVirtualAddrBist(val, offsetStart, offsetEnd);
}

inline uint64_t GetVirtualAddrGroupIndex(uint64_t val) {
    constexpr uint64_t groupIndexStart = 58UL; 
    constexpr uint64_t groupIndexEnd = 59UL; 
    return GetVirtualAddrBist(val, groupIndexStart, groupIndexEnd);
}

inline uint64_t GetVirtaulAddrMemType(uint64_t val) {
    constexpr uint64_t memTypeStart = 60UL; 
    constexpr uint64_t memTypeEnd = 61UL; 
    return GetVirtualAddrBist(val, memTypeStart, memTypeEnd);
}

inline uint64_t GetCoa(const uint32_t index, uint64_t* opAttrs, uint64_t* expressionTable) {
    constexpr uint64_t valueLength = 63;
    constexpr uint64_t valueMask = (1UL << valueLength) - 1;
    const uint64_t encodedValue = opAttrs[index];
    const bool isExpression = (encodedValue >> valueLength) & 1;
    const uint64_t decodedValue = encodedValue & valueMask;
    return isExpression ? expressionTable[decodedValue] : decodedValue;
}

inline std::vector<uint32_t> GetCoaVector(const uint32_t baseIndex, const uint32_t dim, uint64_t* opAttrs,
    uint64_t* expressionTable) {
    std::vector<uint32_t> vec(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        vec[i] = GetCoa(baseIndex + i, opAttrs, expressionTable);
    }
    return vec;
}

inline AicpuParamInfo DecodeAicpuCode(const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode) {
    AicpuParamInfo paramInfo;
    int index = 1;
    paramInfo.outIndex = index + 1;

    index = index + aicpuCode[index] + 1;
    paramInfo.inIndex = index + 1;

    index = index + aicpuCode[index] + 1;
    paramInfo.rawShapeIndex = index + 1;
    paramInfo.rawRankShape = aicpuCode[paramInfo.rawShapeIndex + 2]; // ShmemSignal RawShape[ranksize, ranksize, ranksize, row, col], 2表示rankShape的值
    paramInfo.rawShapeRow = aicpuCode[paramInfo.rawShapeIndex + 3]; // ShmemSignal RawShape[ranksize, ranksize, ranksize, row, col], 3表示row的值
    paramInfo.rawShapeCol = aicpuCode[paramInfo.rawShapeIndex + 4]; // ShmemSignal RawShape[ranksize, ranksize, ranksize, row, col], 4表示col的值
    paramInfo.tileShapeIndex = paramInfo.rawShapeIndex + aicpuCode[index] / 2; // 存储了signal_dim * 2个参数, tieShape往后偏移dim位
    paramInfo.tileShapeRow = aicpuCode[paramInfo.tileShapeIndex + 3]; // ShmemSignal Shape[ranksize, ranksize, ranksize, row, col], 3表示row的值
    paramInfo.tileShapeCol = aicpuCode[paramInfo.tileShapeIndex + 4]; // ShmemSignal Shape[ranksize, ranksize, ranksize, row, col], 4表示col的值
    index = index + aicpuCode[index] + 1;
    if (index + 1 < static_cast<int32_t>(aicpuCode.size())) {
        paramInfo.attrIndex = index + 1;
    }
    return paramInfo;
}
}
