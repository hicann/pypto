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
#include "machine/utils/dynamic/dev_encode.h"

namespace npu::tile_fwk::Distributed {
constexpr uint64_t VECTOR_PRE_SIZE = 1024;

struct TensorInfo {
    uint64_t rawAddr;
    uint32_t dim;
    uint64_t rawIndex;
    int32_t expectedSum;
    std::vector<uint32_t> offset;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> rawShape;
    std::vector<uint32_t> dynValidShape;
};

struct AicpuParamInfo {
    int32_t outIndex{0};
    int32_t inIndex{0};
    int32_t attrIndex{0};
};

inline uint64_t GetVirtualAddrBist(uint64_t val, uint64_t start, uint64_t end)
{
    return (((val) >> (start)) & ((1UL << ((end) - (start) + 1UL)) - 1UL));
}

inline uint64_t GetVirtualAddrOffset(uint64_t val)
{
    constexpr uint64_t offsetStart = 0UL; 
    constexpr uint64_t offsetEnd = 57UL; 
    return GetVirtualAddrBist(val, offsetStart, offsetEnd);
}

inline uint64_t GetVirtualAddrGroupIndex(uint64_t val)
{
    constexpr uint64_t groupIndexStart = 58UL; 
    constexpr uint64_t groupIndexEnd = 59UL; 
    return GetVirtualAddrBist(val, groupIndexStart, groupIndexEnd);
}

inline uint64_t GetVirtaulAddrMemType(uint64_t val)
{
    constexpr uint64_t memTypeStart = 60UL; 
    constexpr uint64_t memTypeEnd = 61UL; 
    return GetVirtualAddrBist(val, memTypeStart, memTypeEnd);
}

inline uint64_t GetCoa(const uint32_t index, __gm__ uint64_t* opAttrs, __gm__ uint64_t* expressionTable)
{
    constexpr uint64_t valueLength = 63;
    constexpr uint64_t valueMask = (1UL << valueLength) - 1;
    const uint64_t encodedValue = opAttrs[index];
    const bool isExpression = (encodedValue >> valueLength) & 1;
    const uint64_t decodedValue = encodedValue & valueMask;
    return isExpression ? expressionTable[decodedValue] : decodedValue;
}

inline std::vector<uint32_t> GetCoaVector(const uint32_t baseIndex, const uint32_t dim, __gm__ uint64_t* opAttrs,
    __gm__ uint64_t* expressionTable)
{
    std::vector<uint32_t> vec(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        vec[i] = GetCoa(baseIndex + i, opAttrs, expressionTable);
    }
    return vec;
}

inline AicpuParamInfo DecodeAicpuCode(const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode)
{
    AicpuParamInfo paramInfo;
    int index = 1;
    paramInfo.outIndex = index + 1;

    index = index + aicpuCode[index] + 1;
    paramInfo.inIndex = index + 1;

    index = index + aicpuCode[index] + 1;
    if (index + 1 < static_cast<int32_t>(aicpuCode.size())) {
        paramInfo.attrIndex = index + 1;
    }
    return paramInfo;
}
}
