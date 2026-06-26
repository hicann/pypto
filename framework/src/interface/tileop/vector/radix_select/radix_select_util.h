/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file radix_select_util.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL__H
#if defined(PTO_NPU_ARCH_A5)
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace RadixSelectUtil {
template <size_t size = sizeof(uint8_t)>
struct IntBySize {
    using T = int8_t;
};
template <>
struct IntBySize<sizeof(uint16_t)> {
    using T = int16_t;
};
template <>
struct IntBySize<sizeof(uint32_t)> {
    using T = int32_t;
};
template <>
struct IntBySize<sizeof(uint64_t)> {
    using T = int64_t;
};
template <size_t size = sizeof(uint8_t)>
struct UIntBySize {
    using T = uint8_t;
};
template <>
struct UIntBySize<sizeof(uint16_t)> {
    using T = uint16_t;
};
template <>
struct UIntBySize<sizeof(uint32_t)> {
    using T = uint32_t;
};
template <>
struct UIntBySize<sizeof(uint64_t)> {
    using T = uint64_t;
};
template <typename T>
struct SignByType {
    using U = typename IntBySize<sizeof(T)>::T;
    static constexpr U value = U(1) << (sizeof(U) * 8 - 1);
};
template <typename T, size_t tile = 32ULL / sizeof(T), bool isCol = false, size_t other = 1>
struct RSTile {
    using U = typename std::conditional_t<
        isCol,
        typename pto::Tile<pto::TileType::Vec, T, tile, other, pto::BLayout::ColMajor, -1, -1>,
        typename pto::Tile<pto::TileType::Vec, T, other, tile, pto::BLayout::RowMajor, -1, -1>
    >;
};
template <typename T, size_t tile = 32ULL / sizeof(T), bool isCol = false, size_t other = 1>
__aicore__ inline auto DefineTile(size_t row, size_t col, size_t addr)
{
    typename RSTile<T, tile, isCol, other>::U t(row, col);
    pto::TASSIGN(t, addr);
    return t;
}
template <typename T, size_t tile = 32ULL / sizeof(T), bool isCol = false, size_t other = 1>
__aicore__ inline auto DefineTile(size_t row, size_t col)
{
    return typename RSTile<T, tile, isCol, other>::U(row, col);
}
template <typename T>
__aicore__ inline size_t DefineWorkSpace(size_t& point, size_t size)
{
    size_t result = point;
    point += size * sizeof(T);
    return result;
}
__aicore__ inline constexpr size_t AlignUp(size_t size, size_t align)
{
    if (align == 0) {
        return size;
    }
    return (size + align - 1) / align * align;
}
#define PTO_RS_GET_STRIDE(valName, layout) \
    size_t valName[] = {\
    static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_1ST, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_2ND, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_3RD, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetStrideDim<DIM_4TH, MAX_DIMS>())}
#define PTO_RS_GET_SHAPE(valName, layout) \
    size_t valName[] = {\
    static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_1ST, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_2ND, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_3RD, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>()),\
    static_cast<size_t>(layout.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>())}
#define PTO_RS_PREPARE \
    using SrcDType = typename SRC::Type;\
    using IdxDType = typename IDX::Type;\
    constexpr auto srcTypeSize = sizeof(SrcDType);\
    constexpr auto idxTypeSize = sizeof(IdxDType);\
    using ConvUIntType = typename UIntBySize<srcTypeSize>::T;\
    using ConvIntType = typename IntBySize<srcTypeSize>::T;\
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<SRC, DIM_5TH, MAX_DIMS>();\
    constexpr int64_t cmpSize = (srcTileW > 256 ? srcTileW : 256) / 8;\
    constexpr auto kAlign = AlignUp(k, 128);\
    PTO_RS_GET_SHAPE(srcShape, src);\
    PTO_RS_GET_STRIDE(srcStride, src);\
    PTO_RS_GET_STRIDE(valStride, value);\
    PTO_RS_GET_STRIDE(idxStride, index)
#define PTO_RS_SORT_ADDR_DEFINE \
    size_t number0Addr = DefineWorkSpace<uint16_t>(point, 16);\
    size_t number1Addr = DefineWorkSpace<uint16_t>(point, 16);\
    size_t number2Addr = DefineWorkSpace<uint16_t>(point, 16);\
    size_t number3Addr = DefineWorkSpace<uint16_t>(point, 16);\
    size_t cnt1Addr = DefineWorkSpace<uint32_t>(point, 8);\
    size_t cnt2Addr = DefineWorkSpace<uint32_t>(point, 8);\
    size_t cnt3Addr = DefineWorkSpace<uint32_t>(point, 8);\
    size_t select1Addr = DefineWorkSpace<uint32_t>(point, kAlign);\
    size_t select2Addr = DefineWorkSpace<uint32_t>(point, kAlign);\
    size_t select3Addr = DefineWorkSpace<uint32_t>(point, kAlign);\
    size_t indexAddr = DefineWorkSpace<uint32_t>(point, kAlign)
#define PTO_RS_SORT_TILE_DEFINE(type) \
    auto sortTempInt##type##KTile = DefineTile<int##type##_t>(1, k, sortTmpAddr);\
    auto sortTempInt##type##MaxTile = DefineTile<int##type##_t>(1, kAlign, sortTmpAddr);\
    auto number0UInt##type##Tile = DefineTile<uint##type##_t>(1, 1, number0Addr);\
    auto number1UInt##type##Tile = DefineTile<uint##type##_t>(1, 1, number1Addr);\
    auto number2UInt##type##Tile = DefineTile<uint##type##_t>(1, 1, number2Addr);\
    auto number3UInt##type##Tile = DefineTile<uint##type##_t>(1, 1, number3Addr);\
    auto cnt1UInt32Tile = DefineTile<uint32_t>(1, 1, cnt1Addr);\
    auto cnt2UInt32Tile = DefineTile<uint32_t>(1, 1, cnt2Addr);\
    auto cnt3UInt32Tile = DefineTile<uint32_t>(1, 1, cnt3Addr);\
    auto select1Int32Tile = DefineTile<int32_t>(1, k, select1Addr);\
    auto select2Int32Tile = DefineTile<int32_t>(1, k, select2Addr);\
    auto select3Int32Tile = DefineTile<int32_t>(1, k, select3Addr);\
    auto indexInt32Tile = DefineTile<int32_t>(1, k, indexAddr)
#define PTO_RS_COMMON_TILE_DEFINE \
    auto srcIntTile = DefineTile<ConvIntType>(1, srcShape[4]);\
    auto valIntTile = DefineTile<ConvIntType>(1, k);\
    auto idxTile = DefineTile<IdxDType>(1, k);\
    auto cmpTile = DefineTile<uint8_t>(1, cmpSize, cmpAddr);\
    auto selectInt32GTTile = DefineTile<int32_t>(1, k, selectGTAddr);\
    auto selectInt32EQTile = DefineTile<int32_t>(1, k, selectEQAddr);\
    auto selectCountUInt32GTTile = DefineTile<uint32_t>(1, 1, selectCountGTAddr);\
    auto selectCountUInt32EQTile = DefineTile<uint32_t>(1, 1, selectCountEQAddr);\
    auto rowMinTile = DefineTile<int32_t>(1, 8, rowMinAddr);\
    auto gatherIntTile = DefineTile<int32_t>(1, 8, gatherAddr);\
    auto uselessTile = DefineTile<uint32_t>(1, 1, uselessAddr)

template <pto::CmpMode cmpMode, typename SELECT, typename SRC, typename K, typename COUNT, typename USELESS>
TILEOP void RadixSelectGather(SELECT select, SRC src, K k, COUNT count, USELESS useless)
{
    pto::TGATHER<SELECT, SRC, K, COUNT, USELESS, cmpMode>(select, src, k, count, useless, 0);
}

/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint16/uint32 |
sortTmpAddr       | kAlign          uint16/uint32 |
number0Addr       | 16              uint16        |
number1Addr       | 16              uint16        |
number2Addr       | 16              uint16        |
number3Addr       | 16              uint16        |
cnt1Addr          | 8               uint32        |
cnt2Addr          | 8               uint32        |
cnt3Addr          | 8               uint32        |
select1Addr       | kAlign          uint32        |
select2Addr       | kAlign          uint32        |
select3Addr       | kAlign          uint32        |
indexAddr         | kAlign          uint32        |
*/
template <int64_t bit, typename SRC, typename IDX, typename TMP, typename TMP2, typename SELECT, typename COUNT, typename NUM, typename USELESS>
TILEOP void RadixSelectSortTwoBitCalc(
    SRC src, IDX index, TMP tmp, TMP2 tmp2, USELESS useless, SELECT select1, SELECT select2, SELECT select3,
    COUNT count1, COUNT count2, COUNT count3, NUM num0, NUM num1, NUM num2, NUM num3)
{
    pto::TSHRS(tmp, src, bit);
    pto::TANDS(tmp, tmp, 0x3);
    RadixSelectGather<pto::CmpMode::EQ>(select2, tmp2, num3, count2, useless);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp2, num2, count1, useless);
    pto::TADD(count3, count1, count2);
    pto::TCONCAT(select3, select2, select1, count2, count1);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp2, num1, count1, useless);
    pto::TADD(count2, count1, count3);
    pto::TCONCAT(select2, select3, select1, count3, count1);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp2, num0, count1, useless);
    if constexpr (bit == 0) {
        pto::TCONCAT(index, select2, select1, count2, count1);
        pto::TGATHER(tmp, src, index, useless);
    } else {
        pto::TCONCAT(select3, select2, select1, count2, count1);
        pto::TGATHER(select1, index, select3, useless);
        pto::TMOV(index, select1);
        pto::TGATHER(tmp, src, select3, useless);
    }
    pto::TMOV(src, tmp);
}

template <int64_t bit, int64_t lastBit, typename SRC, typename IDX, typename TMP, typename TMP2, typename SELECT, typename COUNT, typename NUM, typename USELESS>
TILEOP void RadixSelectSortTwoBit(
    SRC src, IDX index, TMP tmp, TMP2 tmp2, USELESS useless, SELECT select1, SELECT select2, SELECT select3,
    COUNT count1, COUNT count2, COUNT count3, NUM num0, NUM num1, NUM num2, NUM num3)
{
    if constexpr (bit < lastBit) {
        RadixSelectSortTwoBitCalc<bit>(src, index, tmp, tmp2, useless, select1, select2, select3, count1, count2, count3, num0, num1, num2, num3);
        RadixSelectSortTwoBit<bit + 2, lastBit>(src, index, tmp, tmp2, useless, select1, select2, select3, count1, count2, count3, num0, num1, num2, num3);
    }
}

template <typename TMP, typename NUM>
TILEOP void RadixSelectSortPrepare(TMP tmp, NUM num0, NUM num1, NUM num2, NUM num3)
{
    pto::TEXPANDS(tmp, 0x7fff);
    pto::TEXPANDS(num0, 0);
    pto::TEXPANDS(num1, 1);
    pto::TEXPANDS(num2, 2);
    pto::TEXPANDS(num3, 3);
}

template <
    bool isLargest, bool in, bool isUInt, bool isFloat,
    typename SrcDType, typename TWI, typename SRC, typename TMP, typename CMP, typename USELESS>
TILEOP void RadixSelectTwiddle(TWI twi, SRC src, TMP tmp, CMP cmp, USELESS useless)
{
    if constexpr (!isLargest && !in) {
        pto::TNOT(src, src);
    }
    constexpr auto SIGN = SignByType<SrcDType>::value;
    if constexpr (isFloat) {
        constexpr pto::CmpMode cmpMode = in ? pto::CmpMode::LT : pto::CmpMode::GE;
        pto::TCMPS(cmp, src, 0, cmpMode);
        pto::TXORS(tmp, src, SIGN, useless);
        pto::TNOT(twi, src);
        pto::TSEL(twi, cmp, twi, tmp, useless);
    } else if constexpr (isUInt) {
        pto::TMOV(twi, src);
    } else {
        pto::TXORS(twi, src, SIGN, useless);
    }
    if constexpr (!isLargest && in) {
        pto::TNOT(twi, twi);
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL__H