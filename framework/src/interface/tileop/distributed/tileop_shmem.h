/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tileop_shmem.h
 * \brief Shmem (shared memory) tileops: clear/set, GM/UB copy, Put/Get/Signal, Reduce.
 */

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "common.h"
#include <type_traits>
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#ifdef SUPPORT_TILE_TENSOR
#include "pto/comm/pto_comm_inst.hpp"
#endif

namespace TileOp::Distributed {

// ---------------------------------------------------------------------------
// Helper macro for getting GM layout info (stride and offset)
// ---------------------------------------------------------------------------
#define GET_GM_LAYOUT_INFO(tensor, layoutName, s0, s1, s2, off, coordType, coord) \
    const auto layoutName = tensor.GetLayout();                                   \
    auto s0 = layoutName.template GetStrideDim<DIM_1ST, MAX_DIMS>();              \
    auto s1 = layoutName.template GetStrideDim<DIM_2ND, MAX_DIMS>();              \
    auto s2 = layoutName.template GetStrideDim<DIM_3RD, MAX_DIMS>();              \
    auto off = layoutName.template GetGmOffset<coordType, MAX_DIMS>(coord)

// ---------------------------------------------------------------------------
// Shmem tensor/tile type aliases
// ---------------------------------------------------------------------------
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

TILEOP inline ShapeDyn MakeShape(uint32_t row, uint32_t col) { return ShapeDyn(1, 1, 1, row, col); }
TILEOP inline StrideDyn MakeStride(uint32_t row, uint32_t stride) { return StrideDyn(row, row, row, stride, 1); }
TILEOP inline uint32_t ToggleEvent(uint32_t eventId) { return eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0; }

template <typename T>
TILEOP constexpr T CeilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

template <AtomicType atomicType, typename TileType, typename GlobalType>
TILEOP void AtomicStore(GlobalType& global, TileType& tile)
{
    if constexpr (atomicType == AtomicType::ADD) {
        pto::TSTORE<TileType, GlobalType, pto::AtomicType::AtomicAdd>(global, tile);
    } else {
        pto::TSTORE<TileType, GlobalType, pto::AtomicType::AtomicNone>(global, tile);
    }
}

template <typename T>
using ShmemGlobalTensor = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, uint32_t MaxRowShape, uint32_t MaxColShape>
using ShmemUbTile = pto::Tile<
    pto::TileType::Vec, T, MaxRowShape, AlignUp<uint32_t>(MaxColShape * sizeof(T), COPY_BLOCK_BYTE_SIZE) / sizeof(T),
    pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>;

// ---------------------------------------------------------------------------
// Shmem clear / set
// ---------------------------------------------------------------------------
// Zero a shmem region; use PTO TEXPANDS instead of vector_dup for forward compatibility.
// V→MTE3 sync ensures fill completes before TSTORE.
template <typename T, uint32_t bufferEleNum>
TILEOP void ShmemClear(__ubuf__ T* buffer, __gm__ T* shmemTensorAddr, uint32_t shmemTensorEleNum)
{
    ShmemUbTile<T, 1, bufferEleNum> ubTile(1, bufferEleNum);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TEXPANDS(ubTile, static_cast<T>(0));
    PIPE_SYNC_EVENT(PIPE_V, PIPE_MTE3, EVENT_ID0);

    uint32_t fullChunkCount = shmemTensorEleNum / bufferEleNum;

    for (int32_t i = 0; i < fullChunkCount; i++) {
        __gm__ T* dstAddr = shmemTensorAddr + bufferEleNum * i;
        ShapeDyn shape = MakeShape(1, bufferEleNum);
        StrideDyn strideDyn = MakeStride(1, bufferEleNum);
        ShmemGlobalTensor<T> gmTensor(dstAddr, shape, strideDyn);
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicNone>(gmTensor, ubTile);
    }

    uint32_t tailEleNum = shmemTensorEleNum % bufferEleNum;
    if (tailEleNum != 0) {
        __gm__ T* tailDstAddr = shmemTensorAddr + bufferEleNum * fullChunkCount;
        ShapeDyn tailShape = MakeShape(1, tailEleNum);
        StrideDyn tailStrideDyn = MakeStride(1, tailEleNum);
        ShmemGlobalTensor<T> tailGmTensor(tailDstAddr, tailShape, tailStrideDyn);
        ShmemUbTile<T, 1, bufferEleNum> tailUbTile(1, tailEleNum);
        pto::TASSIGN(tailUbTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TSTORE<decltype(tailUbTile), decltype(tailGmTensor), pto::AtomicType::AtomicNone>(
            tailGmTensor, tailUbTile);
    }
}

template <typename T, uint32_t bufferEleNum>
TILEOP void ShmemClearFlag(
    __ubuf__ T* buffer, __gm__ T* shmemTensorAddr, uint32_t worldSize, uint32_t maxTileNume, uint32_t stride)
{
    ShmemUbTile<T, 1, bufferEleNum> ubTile(1, bufferEleNum);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TEXPANDS(ubTile, static_cast<T>(0));
    PIPE_SYNC_EVENT(PIPE_V, PIPE_MTE3, EVENT_ID0);

    uint32_t shmemTensorEleNum = worldSize * maxTileNume;
    uint32_t fullChunkCount = shmemTensorEleNum / bufferEleNum;

    for (int32_t i = 0; i < fullChunkCount; i++) {
        __gm__ T* dstAddr = shmemTensorAddr + bufferEleNum * i;
        ShapeDyn shape = MakeShape(1, bufferEleNum);
        StrideDyn strideDyn = MakeStride(1, bufferEleNum);
        ShmemGlobalTensor<T> gmTensor(dstAddr, shape, strideDyn);
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicNone>(gmTensor, ubTile);
    }
}

template <typename T, uint32_t bufferEleNum, typename T1, typename C1>
TILEOP void ShmemSet(
    CoreFuncParam* param, __ubuf__ T* buffer, T1 shmemTensor, C1 coordinate, uint32_t ownerRank,
    __gm__ int64_t* hcclContext)
{
    const auto shmemTensorLayout = shmemTensor.GetLayout();
    auto shmemTensorOffset = shmemTensorLayout.template GetGmOffset<C1, MAX_DIMS>(coordinate);
    auto shape0 = shmemTensorLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = shmemTensorLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = shmemTensorLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = shmemTensorLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = shmemTensorLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    using shmemTensorDtype = typename T1::Type;

    __gm__ shmemTensorDtype* shmemTensorAddr =
        MapVirtualAddr<shmemTensorDtype>(hcclContext, shmemTensor.GetAddr(), ownerRank) + shmemTensorOffset;

    uint32_t shmemTensorEleNum = shape0 * shape1 * shape2 * shape3 * shape4;

    ShmemClear<T, bufferEleNum>(buffer, shmemTensorAddr, shmemTensorEleNum);
}

template <typename T, uint32_t worldSize, uint32_t stride, uint32_t bufferEleNum, typename T1>
TILEOP void ShmemSet(
    CoreFuncParam* param, __ubuf__ T* buffer, T1 shmemTensor, uint32_t ownerRank, __gm__ int64_t* hcclContext)
{
    uint32_t maxTileNume =
        static_cast<uint32_t>(TileOp::Distributed::DecodeShmemAddrMaxTileNum((uint64_t)shmemTensor.GetAddr()));

    __gm__ T* shmemTensorAddr = MapVirtualAddr<T>(hcclContext, shmemTensor.GetAddr(), ownerRank);

    ShmemClearFlag<T, bufferEleNum>(buffer, shmemTensorAddr, worldSize, maxTileNume, stride);
}

// ---------------------------------------------------------------------------
// Copy: GM↔GM (via UB, with optional type conversion and ping-pong)
// ---------------------------------------------------------------------------
template <
    typename TargetType, typename UBType, typename SourceType, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlock(
    __gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source, uint32_t rowShape, uint32_t colShape,
    uint32_t eventId = EVENT_ID0)
{
    wait_flag(PIPE_MTE3, PIPE_S, eventId);
    PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE2, eventId);

    ShapeDyn shape = MakeShape(rowShape, colShape);
    StrideDyn srcStrideDyn = MakeStride(rowShape, srcStride);
    StrideDyn dstStrideDyn = MakeStride(rowShape, dstStride);
    ShmemGlobalTensor<SourceType> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<TargetType> dstGlobal(target, shape, dstStrideDyn);
    constexpr uint64_t copyLen =
        bufferRowShape * AlignUp<uint64_t>(bufferColShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ float* castUb = (__ubuf__ float*)(buffer + copyLen);
    constexpr bool kAtomicAdd = (atomicType == AtomicType::ADD);
    using SrcElemType = std::conditional_t<kAtomicAdd, UBType, float>;
    using DstElemType = std::conditional_t<kAtomicAdd, float, UBType>;
    ShmemUbTile<SrcElemType, bufferRowShape, bufferColShape> srcTile(rowShape, colShape);
    ShmemUbTile<DstElemType, bufferRowShape, bufferColShape> dstTile(rowShape, colShape);
    if constexpr (kAtomicAdd) {
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(castUb));
    } else {
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(buffer));
    }
    pto::TLOAD(srcTile, srcGlobal);
    PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_V, eventId);
    pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
    PIPE_SYNC_EVENT(PIPE_V, PIPE_MTE3, eventId);
    AtomicStore<atomicType>(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, eventId);
}

template <
    typename TargetType, typename UBType, typename SourceType, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmRow(
    __gm__ TargetType* dstPtr, __gm__ SourceType* srcPtr, __ubuf__ UBType* bufferA, __ubuf__ UBType* bufferB,
    uint32_t rowShape, uint32_t colTailShape, uint32_t colFullBlockCount, uint32_t& eventId)
{
    uint32_t colOffset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, colOffset += bufferColShape) {
        __ubuf__ UBType* useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
        CopyGmToGmBlock<
            TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
            dstPtr + colOffset, useBuffer, srcPtr + colOffset, rowShape, bufferColShape, eventId);
        eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
    }
    if (colTailShape > 0) {
        __ubuf__ UBType* useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
        CopyGmToGmBlock<
            TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
            dstPtr + colOffset, useBuffer, srcPtr + colOffset, rowShape, colTailShape, eventId);
        eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
    }
}

template <
    bool useTPut, typename DataType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType = AtomicType::SET>
TILEOP void CopyGmToGmByPutGet(
    __gm__ DataType* target, __ubuf__ DataType* buffer, __gm__ DataType* source, uint32_t validRowShape,
    uint32_t validColShape)
{
    static_assert(bufferRowShape > 0, "bufferRowShape must be greater than 0.");
    constexpr uint32_t kMaxTileRows = 4095;
    constexpr uint32_t kEffectiveRows = bufferRowShape < kMaxTileRows ? bufferRowShape : kMaxTileRows;
    constexpr uint32_t kChunkRows = tileRowShape < kEffectiveRows ? tileRowShape : kEffectiveRows;

    constexpr uint32_t kAlignedCols =
        AlignUp<uint32_t>(tileColShape * sizeof(DataType), COPY_BLOCK_BYTE_SIZE) / sizeof(DataType);
    constexpr uint32_t kHalfBufferEleCount = bufferRowShape * kAlignedCols;

    ShapeDyn shape = MakeShape(validRowShape, validColShape);
    StrideDyn srcStrideDyn = MakeStride(validRowShape, srcStride);
    StrideDyn dstStrideDyn = MakeStride(validRowShape, dstStride);
    ShmemGlobalTensor<DataType> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<DataType> dstGlobal(target, shape, dstStrideDyn);

    ShmemUbTile<DataType, kChunkRows, tileColShape> pingTile(kChunkRows, validColShape);
    ShmemUbTile<DataType, kChunkRows, tileColShape> pongTile(kChunkRows, validColShape);
    pto::TASSIGN(pingTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TASSIGN(pongTile, reinterpret_cast<uintptr_t>(buffer + kHalfBufferEleCount));
    if constexpr (useTPut) {
        if constexpr (atomicType == AtomicType::ADD) {
            pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstGlobal, srcGlobal, pingTile, pongTile);
        } else {
            pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstGlobal, srcGlobal, pingTile, pongTile);
        }
    } else {
        pto::comm::TGET(dstGlobal, srcGlobal, pingTile, pongTile);
    }
    PIPE_SYNC_EVENT(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

// Full tile copy with row/column chunking. Ping-pong layout: same type bufferA|bufferB; with conversion
// bufferA|castUbA|bufferB|castUbB.
template <
    typename TargetType, typename UBType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape,
    uint32_t bufferRowShape, uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmByLaodStore(
    __gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source, uint32_t validRowShape,
    uint32_t validColShape)
{
    uint32_t rowFullBlockCount = validRowShape / bufferRowShape;
    uint32_t colFullBlockCount = validColShape / bufferColShape;
    uint32_t rowTailShape = validRowShape % bufferRowShape;
    uint32_t colTailShape = validColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);

    uint32_t eventId = EVENT_ID0;

    constexpr uint32_t copyLen =
        bufferRowShape * AlignUp<uint32_t>(bufferColShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ UBType* bufferA = buffer;
    __ubuf__ UBType* bufferB = buffer + copyLen;

    if constexpr (!std::is_same_v<TargetType, SourceType>) {
        constexpr uint64_t castSize = AlignUp<uint64_t>(copyLen * sizeof(float), 256);
        bufferB = buffer + copyLen + castSize / sizeof(UBType);
    }

    __gm__ SourceType* srcPtr = source;
    __gm__ TargetType* dstPtr = target;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount;
         ++rowIndex, srcPtr += srcRowStride, dstPtr += dstRowStride) {
        CopyGmToGmRow<TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
            dstPtr, srcPtr, bufferA, bufferB, bufferRowShape, colTailShape, colFullBlockCount, eventId);
    }

    if (rowTailShape > 0) {
        CopyGmToGmRow<TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
            dstPtr, srcPtr, bufferA, bufferB, rowTailShape, colTailShape, colFullBlockCount, eventId);
    }

    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
}

// ---------------------------------------------------------------------------
// Copy: GM → UB (single block, optional type conversion)
// ---------------------------------------------------------------------------
template <
    typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape, uint32_t srcStride,
    uint32_t dstStride>
TILEOP void CopyGmToUbBlock(
    uint64_t target, __ubuf__ TargetType* buffer, __gm__ SourceType* source, uint32_t ubValidShape0,
    uint32_t ubValidShape1)
{
    ShapeDyn shape = MakeShape(ubValidShape0, ubValidShape1);
    StrideDyn srcStrideDyn = MakeStride(ubValidShape0, srcStride);
    ShmemGlobalTensor<SourceType> srcGlobal(source, shape, srcStrideDyn);
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE2, EVENT_ID0);
        ShmemUbTile<TargetType, rowShape, colShape> ubTile(ubValidShape0, ubValidShape1);
        pto::TASSIGN(ubTile, target);
        pto::TLOAD(ubTile, srcGlobal);
        PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_S, EVENT_ID0);
    } else {
        __ubuf__ float* castUb = (__ubuf__ float*)buffer;
        ShmemUbTile<float, rowShape, colShape> srcTile(ubValidShape0, ubValidShape1);
        ShmemUbTile<TargetType, rowShape, colShape> dstTile(ubValidShape0, ubValidShape1);
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TASSIGN(dstTile, target);
        pto::TLOAD(srcTile, srcGlobal);
        PIPE_SYNC_EVENT(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
        PIPE_SYNC_EVENT(PIPE_V, PIPE_S, EVENT_ID0);
    }
}

// ---------------------------------------------------------------------------
// Shmem Put / Get / Signal
// ---------------------------------------------------------------------------
// Put: local GM (or inShmem GM) → remote shmem GM.
template <
    typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType, typename T1, typename T2,
    typename C1, typename C2>
TILEOP void ShmemPut(
    CoreFuncParam* param, __ubuf__ NonShmemType* buffer, T1 src, T2 dst, C1 srcCoordinate, C2 dstCoordinate,
    uint32_t srcValidShape0, uint32_t srcValidShape1, uint32_t srcValidShape2, uint32_t srcValidShape3,
    uint32_t srcValidShape4, uint32_t ownerRank, __gm__ int64_t* hcclContext)
{
    static_assert(T1::FORMAT == Hardware::GM && T2::FORMAT == Hardware::GM);
    GET_GM_LAYOUT_INFO(src, srcLayout, srcStride0, srcStride1, srcStride2, srcOffset, C1, srcCoordinate);
    GET_GM_LAYOUT_INFO(dst, dstLayout, dstStride0, dstStride1, dstStride2, dstOffset, C2, dstCoordinate);

    if constexpr (atomicType == AtomicType::ADD) {
        SetAttomicType<ShmemType>();
        set_atomic_add();
    }

    for (LoopVar index0 = 0; index0 < srcValidShape0; ++index0) {
        for (LoopVar index1 = 0; index1 < srcValidShape1; ++index1) {
            for (LoopVar index2 = 0; index2 < srcValidShape2; ++index2) {
                auto srcOffset3d = index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                auto dstOffset3d = index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;
                __gm__ NonShmemType* srcAddr = src.GetAddr() + srcOffset + srcOffset3d;
                __gm__ ShmemType* dstAddr =
                    MapVirtualAddr<ShmemType>(hcclContext, dst.GetAddr(), ownerRank) + dstOffset + dstOffset3d;
                if constexpr (std::is_same_v<ShmemType, NonShmemType>) {
                    CopyGmToGmByPutGet<
                        true, ShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride, atomicType>(
                        dstAddr, buffer, srcAddr, srcValidShape3, srcValidShape4);
                } else {
                    CopyGmToGmByLaodStore<
                        ShmemType, NonShmemType, NonShmemType, tileRowShape, tileColShape, bufferRowShape,
                        bufferColShape, srcStride, dstStride, atomicType>(
                        dstAddr, buffer, srcAddr, srcValidShape3, srcValidShape4);
                }
            }
        }
    }

    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template <
    typename InShmemType, typename OutShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(
    CoreFuncParam* param, __ubuf__ InShmemType* buffer, __gm__ InShmemType* inShmemDataBaseAddr,
    __gm__ OutShmemType* shmemDataBaseAddr, uint32_t inShmemDataOffset0, uint32_t inShmemDataOffset1,
    uint32_t inShmemDataOffset2, uint32_t inShmemDataRawShape0, uint32_t inShmemDataRawShape1,
    uint32_t inShmemDataRawShape2, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t validShape0,
    uint32_t validShape1, __gm__ int64_t* hcclContext)
{
    (void)inShmemDataRawShape0;
    (void)shmemDataRawShape0;

    __gm__ InShmemType* inShmemDataAddr =
        MapVirtualAddr<InShmemType>(hcclContext, inShmemDataBaseAddr, inShmemDataOffset0) +
        CalcLinearOffset(inShmemDataRawShape2, inShmemDataOffset1, inShmemDataOffset2);
    __gm__ OutShmemType* shmemDataAddr =
        MapVirtualAddr<OutShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        CalcLinearOffset(shmemDataRawShape2, shmemDataOffset1, shmemDataOffset2);

    if constexpr (std::is_same_v<OutShmemType, InShmemType>) {
        CopyGmToGmByPutGet<
            true, OutShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride, atomicType>(
            shmemDataAddr, buffer, inShmemDataAddr, validShape0, validShape1);
    } else {
        CopyGmToGmByLaodStore<
            OutShmemType, InShmemType, InShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape,
            srcStride, dstStride, atomicType>(shmemDataAddr, buffer, inShmemDataAddr, validShape0, validShape1);
    }
}

// Put UB directly to remote shmem GM.
template <
    typename Type, uint32_t tileRowShape, uint32_t tileColShape, uint32_t dstStride, AtomicType atomicType,
    typename T1, typename T2, typename C1, typename C2>
TILEOP void ShmemPutUb2Gm(
    CoreFuncParam* param, T1 src, T2 dst, C1 srcCoordinate, C2 dstCoordinate, uint32_t outValidShape0,
    uint32_t outValidShape1, uint32_t outValidShape2, uint32_t outValidShape3, uint32_t outValidShape4,
    uint32_t ownerRank, __gm__ int64_t* hcclContext)
{
    static_assert(T1::FORMAT == Hardware::UB && T2::FORMAT == Hardware::GM);
    GET_GM_LAYOUT_INFO(src, srcLayout, srcStride0, srcStride1, srcStride2, srcOffset, C1, srcCoordinate);
    GET_GM_LAYOUT_INFO(dst, dstLayout, dstStride0, dstStride1, dstStride2, dstOffset, C2, dstCoordinate);

    for (LoopVar index0 = 0; index0 < outValidShape0; ++index0) {
        for (LoopVar index1 = 0; index1 < outValidShape1; ++index1) {
            for (LoopVar index2 = 0; index2 < outValidShape2; ++index2) {
                auto srcOffset3d = index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                auto dstOffset3d = index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;
                uint64_t srcAddr = src.GetAddr() + srcOffset + srcOffset3d;
                __gm__ Type* dstAddr =
                    MapVirtualAddr<Type>(hcclContext, dst.GetAddr(), ownerRank) + dstOffset + dstOffset3d;

                PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE3, EVENT_ID0);

                ShapeDyn shape = MakeShape(outValidShape3, outValidShape4);
                StrideDyn dstStrideDyn = MakeStride(outValidShape3, dstStride);

                ShmemGlobalTensor<Type> dstGlobal(dstAddr, shape, dstStrideDyn);
                ShmemUbTile<Type, tileRowShape, tileColShape> ubTile(outValidShape3, outValidShape4);
                pto::TASSIGN(ubTile, srcAddr);

                AtomicStore<atomicType>(dstGlobal, ubTile);

                PIPE_SYNC_EVENT(PIPE_MTE3, PIPE_S, EVENT_ID0);
            }
        }
    }
}

// Signal: write value to remote ranks; S→MTE3 sync so scalar write is visible to TSTORE.
template <
    int64_t value, int32_t stride, AtomicType atomicType, bool notifyAll, uint32_t worldSize, int32_t tileShape0,
    int32_t tileShape1, int32_t tileShape2, int32_t tileShape3, int32_t tileShapeDim, typename T1, typename C1>
TILEOP void ShmemSignal(
    CoreFuncParam* param, __ubuf__ int32_t* buffer, T1 src, C1 srcCoordinate, uint32_t ownerRank,
    __gm__ int64_t* hcclContext)
{
    const auto srcLayout = src.GetLayout();
    int32_t srcShape1 = static_cast<int32_t>(srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>());
    int32_t srcShape2 = static_cast<int32_t>(srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>());
    int32_t srcShape3 = static_cast<int32_t>(srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>());
    int32_t srcShape4 = static_cast<int32_t>(srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());
    int32_t c0 = static_cast<int32_t>(TileOp::GetTupleElement<C1, DIM_1ST, MAX_DIMS, 0>(srcCoordinate));
    int32_t c1 = static_cast<int32_t>(TileOp::GetTupleElement<C1, DIM_2ND, MAX_DIMS, 0>(srcCoordinate));
    int32_t c2 = static_cast<int32_t>(TileOp::GetTupleElement<C1, DIM_3RD, MAX_DIMS, 0>(srcCoordinate));
    int32_t c3 = static_cast<int32_t>(TileOp::GetTupleElement<C1, DIM_4TH, MAX_DIMS, 0>(srcCoordinate));
    int32_t c4 = static_cast<int32_t>(TileOp::GetTupleElement<C1, DIM_5TH, MAX_DIMS, 0>(srcCoordinate));

    uint32_t tileIndex = 0;
    uint32_t multiplier = 1;
    uint32_t totalTileNum = 1;

    int32_t srcShapes[] = {0, srcShape1, srcShape2, srcShape3, srcShape4};
    int32_t coords[] = {c0, c1, c2, c3, c4};

    constexpr int32_t startDim = MAX_DIMS - tileShapeDim;
    constexpr int32_t tileShape[] = {tileShape0, tileShape1, tileShape2, tileShape3};

    for (uint32_t dimIdx = 0; dimIdx < tileShapeDim; ++dimIdx) {
        int32_t curDim = startDim + static_cast<int32_t>(dimIdx);
        int32_t rawShape = srcShapes[curDim];
        int32_t tileShapeVal = tileShape[dimIdx];
        int32_t offset = coords[curDim];

        int32_t tileNum = CeilDiv(rawShape, tileShapeVal);
        int32_t dimTileIdx = offset / tileShapeVal;

        tileIndex += static_cast<uint32_t>(dimTileIdx) * multiplier;
        multiplier *= static_cast<uint32_t>(tileNum);
        totalTileNum *= static_cast<uint32_t>(tileNum);
    }

    buffer[0] = static_cast<int32_t>(value);
    constexpr uint32_t signalColShape = 8; // 8 * 4=32B alignment
    ShmemUbTile<int32_t, 1, signalColShape> signalTile(1, 1);
    pto::TASSIGN(signalTile, reinterpret_cast<uintptr_t>(buffer));

    PIPE_SYNC_EVENT(PIPE_S, PIPE_MTE3, EVENT_ID0);

    ShapeDyn signalShape = MakeShape(1, 1);
    StrideDyn signalStride = MakeStride(1, 1);

    uint32_t sRank = notifyAll ? 0 : ownerRank;
    uint32_t eRank = notifyAll ? worldSize : sRank + 1;
    uint64_t baseOffset =
        static_cast<uint64_t>(CalcLinearOffset(totalTileNum, coords[startDim - 1], tileIndex) * stride);
    for (uint32_t rankId = sRank; rankId < eRank; rankId++) {
        __gm__ int32_t* shmemSignalAddr = MapVirtualAddr<int32_t, 1>(hcclContext, src.GetAddr(), rankId) + baseOffset;
        ShmemGlobalTensor<int32_t> signalGlobal(shmemSignalAddr, signalShape, signalStride);
        AtomicStore<atomicType>(signalGlobal, signalTile);
    }
}

// Get: remote shmem GM → local GM.
template <
    typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType, typename T1, typename T2,
    typename C1, typename C2>
TILEOP void ShmemGet(
    CoreFuncParam* param, __ubuf__ NonShmemType* buffer, T1 dst, T2 src, C1 dstCoordinate, C2 srcCoordinate,
    uint32_t dstValidShape0, uint32_t dstValidShape1, uint32_t dstValidShape2, uint32_t dstValidShape3,
    uint32_t dstValidShape4, uint32_t ownerRank, __gm__ int64_t* hcclContext)
{
    static_assert(T1::FORMAT == Hardware::GM && T2::FORMAT == Hardware::GM);
    GET_GM_LAYOUT_INFO(src, srcLayout, srcStride0, srcStride1, srcStride2, srcOffset, C2, srcCoordinate);
    GET_GM_LAYOUT_INFO(dst, dstLayout, dstStride0, dstStride1, dstStride2, dstOffset, C1, dstCoordinate);

    for (LoopVar index0 = 0; index0 < dstValidShape0; ++index0) {
        for (LoopVar index1 = 0; index1 < dstValidShape1; ++index1) {
            for (LoopVar index2 = 0; index2 < dstValidShape2; ++index2) {
                auto srcOffset3d = index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                auto dstOffset3d = index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;
                __gm__ ShmemType* srcAddr =
                    MapVirtualAddr<ShmemType>(hcclContext, src.GetAddr(), ownerRank) + srcOffset + srcOffset3d;
                __gm__ NonShmemType* dstAddr = dst.GetAddr() + dstOffset + dstOffset3d;

                if constexpr (std::is_same_v<NonShmemType, ShmemType>) {
                    CopyGmToGmByPutGet<
                        false, NonShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride>(
                        dstAddr, buffer, srcAddr, dstValidShape3, dstValidShape4);
                } else {
                    CopyGmToGmByLaodStore<
                        NonShmemType, NonShmemType, ShmemType, tileRowShape, tileColShape, bufferRowShape,
                        bufferColShape, srcStride, dstStride, atomicType>(
                        dstAddr, buffer, srcAddr, dstValidShape3, dstValidShape4);
                }
            }
        }
    }
}

// Get: remote shmem GM → UB (single block, optional type conversion).
template <
    typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType, typename T1, typename T2,
    typename C1, typename C2>
TILEOP void ShmemGetGm2Ub(
    CoreFuncParam* param, __ubuf__ UBType* buffer, T1 dst, T2 src, C1 dstCoordinate, C2 srcCoordinate,
    uint32_t dstValidShape0, uint32_t dstValidShape1, uint32_t dstValidShape2, uint32_t dstValidShape3,
    uint32_t dstValidShape4, uint32_t ownerRank, __gm__ int64_t* hcclContext)
{
    static_assert(T1::FORMAT == Hardware::UB && T2::FORMAT == Hardware::GM);
    GET_GM_LAYOUT_INFO(src, srcLayout, srcStride0, srcStride1, srcStride2, srcOffset, C2, srcCoordinate);
    GET_GM_LAYOUT_INFO(dst, dstLayout, dstStride0, dstStride1, dstStride2, dstOffset, C1, dstCoordinate);

    for (LoopVar index0 = 0; index0 < dstValidShape0; ++index0) {
        for (LoopVar index1 = 0; index1 < dstValidShape1; ++index1) {
            for (LoopVar index2 = 0; index2 < dstValidShape2; ++index2) {
                auto srcOffset3d = index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                auto dstOffset3d = index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;

                __gm__ ShmemType* srcAddr =
                    MapVirtualAddr<ShmemType>(hcclContext, src.GetAddr(), ownerRank) + srcOffset + srcOffset3d;
                uint64_t dstAddr = dst.GetAddr() + dstOffset + dstOffset3d;

                CopyGmToUbBlock<UBType, ShmemType, bufferRowShape, bufferColShape, srcStride, dstStride>(
                    dstAddr, buffer, srcAddr, dstValidShape3, dstValidShape4);
            }
        }
    }
}

} // namespace TileOp::Distributed

#endif
