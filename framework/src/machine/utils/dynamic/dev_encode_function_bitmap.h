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
 * \file dev_encode_function_bitmap.h
 * \brief Inline bitmap operations for DevAscendFunction.
 */

#pragma once

#include "dev_encode_function.h"

namespace npu::tile_fwk::dynamic {

inline bool DevAscendFunction::IsDeadEndHub(uint32_t opIndex) const
{
    if (deadEndHubBitmap_.size() == 0)
        return false;
    uint32_t wordIdx = opIndex / 64;
    return (At(deadEndHubBitmap_, wordIdx) & (1ULL << (opIndex % 64))) != 0;
}

inline bool DevAscendFunction::IsTailTask(uint32_t opIndex) const
{
    if (tailTaskBitmap_.size() == 0) {
        return false;
    }
    uint32_t wordIdx = opIndex / 64;
    return (At(tailTaskBitmap_, wordIdx) & (1ULL << (opIndex % 64))) != 0;
}

inline void DevAscendFunction::ClearTailTask(uint32_t opIndex)
{
    if (tailTaskBitmap_.size() == 0) {
        return;
    }
    uint32_t wordIdx = opIndex / 64;
    At(tailTaskBitmap_, wordIdx) &= ~(1ULL << (opIndex % 64));
}

inline bool DevAscendFunction::ClearDeadEndHub(uint32_t opIndex)
{
    if (deadEndHubBitmap_.size() == 0)
        return false;
    uint32_t wordIdx = opIndex / 64;
    uint64_t bit = 1ULL << (opIndex % 64);
    auto& word = At(deadEndHubBitmap_, wordIdx);
    bool wasSet = (word & bit) != 0;
    word &= ~bit;
    return wasSet;
}

inline void DevAscendFunction::PropagateDeadHubClear(uint32_t clearedOpIdx)
{
    constexpr size_t kMaxStack = 128;
    uint32_t stack[kMaxStack];
    size_t stackSize = 0;
    stack[stackSize++] = clearedOpIdx;

    size_t opCount = GetOperationSize();
    while (stackSize > 0) {
        uint32_t target = stack[--stackSize];
        for (size_t op = 0; op < opCount; op++) {
            size_t succSize;
            const int* succList = GetOperationDepGraphSuccAddr(static_cast<int>(op), succSize);
            for (size_t j = 0; j < succSize; j++) {
                if (static_cast<uint32_t>(succList[j]) != target)
                    continue;
                ClearTailTask(static_cast<uint32_t>(op));
                if (ClearDeadEndHub(static_cast<uint32_t>(op)) && stackSize < kMaxStack) {
                    stack[stackSize++] = static_cast<uint32_t>(op);
                }
                break;
            }
        }
    }
}

inline size_t DevAscendFunction::GetBitmapByteSize() const { return deadEndHubBitmap_.size() * sizeof(uint64_t); }

inline void DevAscendFunction::BackupBitmapTo(uint64_t* deadEndBuf, uint64_t* tailBuf, size_t byteSize) const
{
    if (byteSize == 0) {
        return;
    }
    DevMemcpyS(deadEndBuf, byteSize, &At(deadEndHubBitmap_, 0), byteSize);
    DevMemcpyS(tailBuf, byteSize, &At(tailTaskBitmap_, 0), byteSize);
}

inline void DevAscendFunction::RestoreBitmapFrom(const uint64_t* deadEndBuf, const uint64_t* tailBuf, size_t byteSize)
{
    if (byteSize == 0) {
        return;
    }
    DevMemcpyS(&At(deadEndHubBitmap_, 0), byteSize, deadEndBuf, byteSize);
    DevMemcpyS(&At(tailTaskBitmap_, 0), byteSize, tailBuf, byteSize);
}

} // namespace npu::tile_fwk::dynamic
