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
 * \file moe_combine.h
 * \brief
*/

#ifndef __DISTRIBUTED_COMBINE__
#define __DISTRIBUTED_COMBINE__

#include "common.h"
#include "hccl_context.h"

#include <type_traits>

namespace TileOp::Distributed {
template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineSend(__gm__ int32_t* dummyOut, __ubuf__ T* dataBuffer, __ubuf__ int32_t* combineInfoBuffer,
    __ubuf__ int32_t* signalBuffer, __gm__ T* in, __gm__ int32_t* combineInfo, __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr, uint64_t inOffset0, uint64_t inOffset1, __gm__ int64_t* hcclContext)
{
    (void)dummyOut;
    (void)inOffset1;

    vector_dup(signalBuffer, 0, 1, 1, 1, 8, 0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    signalBuffer[0] = 1;

    for (uint64_t row = inOffset0; row < inOffset0 + rowShape; row++) {
        TileOp::UBCopyIn<int32_t, 1, 3, 8, 3>(combineInfoBuffer, combineInfo + MOE_COMBINE_INFO_NUM * row);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int32_t rankId = combineInfoBuffer[0];
        if (rankId == -1) {
            break;
        }
        int32_t tokenId = combineInfoBuffer[1];
        int32_t kOffset = combineInfoBuffer[2];

        __gm__ T* winDataAddr = MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, rankId) +
            colShape * (topK * tokenId + kOffset);
        __gm__ int32_t* winSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, colShape, paddedColShape, colShape>(dataBuffer, in + colShape * row);

        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(winDataAddr, dataBuffer);
        pipe_barrier(PIPE_MTE3);

        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        set_atomic_add();
        set_atomic_s32();
        copy_ubuf_to_gm(winSignalAddr, signalBuffer, 0, 1, 1, 0, 0);
        set_atomic_none();
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

TILEOP void ShmemMoeCombineWaitSignal(__gm__ int32_t* winSignalAddr, __ubuf__ int32_t* signalBuffer,
    int32_t expectedValue)
{
    do {
        copy_gm_to_ubuf(signalBuffer, winSignalAddr, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    } while (signalBuffer[0] != expectedValue);
}

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineComputeRoutedExperts(__ubuf__ T* out, __ubuf__ float* mulFp32Buffer,
    __ubuf__ float* sumFp32Buffer, __ubuf__ float* scale, __gm__ T* winDataAddr, uint8_t repeat)
{
    for (int kOffset = 0; kOffset < topK; kOffset++) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, colShape, paddedColShape, colShape>(out, winDataAddr);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(mulFp32Buffer, out, repeat, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmuls(mulFp32Buffer, mulFp32Buffer, static_cast<float>(scale[kOffset]), repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(sumFp32Buffer, mulFp32Buffer, sumFp32Buffer, repeat, 1, 1, 1, 8, 8, 8);
        winDataAddr += colShape;
    }
}

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineCompute(__ubuf__ T* out, __ubuf__ float* mulFp32Buffer, __ubuf__ float* sumFp32Buffer,
    __ubuf__ float* scale, __gm__ T* winDataAddr)
{
    uint8_t repeat = static_cast<uint8_t>(
        AlignUp<uint16_t>(sizeof(float) * paddedColShape, VECTOR_INSTRUCTION_BYTE_SIZE) / VECTOR_INSTRUCTION_BYTE_SIZE
    );

    vector_dup(sumFp32Buffer, 0.0f, repeat, 1, 1, 8, 8);

    ShmemMoeCombineComputeRoutedExperts<T, topK, colShape, paddedColShape>(out, mulFp32Buffer, sumFp32Buffer, scale,
        winDataAddr, repeat);

    pipe_barrier(PIPE_V);
    vconv_f322bf16a(out, sumFp32Buffer, repeat, 1, 1, 4, 8);
}

template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineReceive(__gm__ T* out, __ubuf__ float* mulFp32Buffer, __ubuf__ float* sumFp32Buffer,
    __ubuf__ T* outBuffer, __gm__ int32_t* dummyIn, __ubuf__ float* scale, __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr, uint64_t shmemDataOffset0, uint64_t shmemDataOffset1,
    uint64_t shmemDataOffset2, uint64_t shmemDataOffset3, __gm__ int64_t* hcclContext)
{
    (void)dummyIn;
    (void)shmemDataOffset1;
    (void)shmemDataOffset3;

    uint64_t thisRankId = shmemDataOffset0;
    uint64_t rowOffset = shmemDataOffset2;

    for (uint64_t tokenId = rowOffset; tokenId < rowOffset + rowShape; tokenId++) {
        __gm__ int32_t* winSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, thisRankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;
        __ubuf__ int32_t* signalBuffer = reinterpret_cast<__ubuf__ int32_t*>(outBuffer);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        ShmemMoeCombineWaitSignal(winSignalAddr, signalBuffer, topK);

        constexpr uint32_t scaleColShape = AlignUp<uint32_t>(sizeof(float) * topK, COPY_BLOCK_BYTE_SIZE) /
            sizeof(float);
        __gm__ T* winDataAddr = MapVirtualAddr<T>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
            colShape * topK * tokenId;
        ShmemMoeCombineCompute<T, topK, colShape, paddedColShape>(outBuffer, mulFp32Buffer,
            sumFp32Buffer, scale + scaleColShape * tokenId, winDataAddr);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(out + colShape * tokenId, outBuffer);
    }
}
} // namespace TileOp::Distributed

#endif