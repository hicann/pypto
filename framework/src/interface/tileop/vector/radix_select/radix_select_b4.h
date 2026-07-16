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
 * \file radix_select_b4.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_B4__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_B4__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_util.h"

namespace RadixSelectUtil {
/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint32  |
srcTwiddleInAddr  | srcTileW        srcType |
cmpAddr           | cmpAlign        uint8   |
high1Addr         | 32              uint8   |
high2Addr         | 32              uint8   |
high3Addr         | 32              uint8   |
selectCountGTAddr | 8               uint32  | and rowMinAddr
selectCountEQAddr | 8               uint32  | and gatherAddr
remindKAddr       | 8               uint32  |
kthValueAddr      | 8               uint32  |
tmpAddr           | 8               uint32  |
uselessAddr       | 8               uint32  |
histogramAddr     | 256             uint32  | selectGTAddr     | kAlign uint32 |
histogramTmpAddr  | 256             uint32  | selectEQAddr     | kAlign uint32 |
*/
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalcB4(VAL value, IDX index, TMP tmp, SRC src)
{
    PTO_RS_PREPARE;
    constexpr bool isUInt = std::is_same_v<SrcDType, uint32_t>;
    constexpr bool isFloat = !(std::is_same_v<SrcDType, uint32_t> || std::is_same_v<SrcDType, int32_t>);
    // Define memory address
    size_t point = tmp.GetAddr();
    size_t srcMaskAddr = DefineWorkSpace<uint32_t>(point, AlignUp(srcTileW, 128));
    size_t sortTmpAddr_ = point;
    size_t srcTwiddleInAddr = DefineWorkSpace<SrcDType>(point, srcTileW);
    size_t cmpAddr = DefineWorkSpace<uint8_t>(point, AlignUp(cmpSize, 32));
    size_t high1Addr = DefineWorkSpace<uint8_t>(point, 32);
    size_t high2Addr = DefineWorkSpace<uint8_t>(point, 32);
    size_t high3Addr = DefineWorkSpace<uint8_t>(point, 32);
    size_t selectCountGTAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t rowMinAddr = selectCountGTAddr;
    size_t selectCountEQAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t gatherAddr = selectCountEQAddr;
    size_t remindKAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t kthValueAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t tmpAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t uselessAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t histogramAddr = DefineWorkSpace<uint32_t>(point, 256);
    size_t histogramTmpAddr = DefineWorkSpace<uint32_t>(point, 256);
    point = histogramAddr;
    size_t selectGTAddr = DefineWorkSpace<uint32_t>(point, kAlign);
    size_t selectEQAddr = DefineWorkSpace<uint32_t>(point, kAlign);
    point = sortTmpAddr_;
    size_t sortTmpAddr = DefineWorkSpace<uint32_t>(point, kAlign);
    PTO_RS_SORT_ADDR_DEFINE;
    // Define tile
    PTO_RS_COMMON_TILE_DEFINE;
    auto twiddleIntTile = DefineTile<ConvIntType>(1, srcShape[4], srcTwiddleInAddr);
    auto twiddleIntKTile = DefineTile<ConvIntType>(1, k, srcTwiddleInAddr);
    auto twiddleUIntTile = DefineTile<ConvUIntType, 32>(1, srcShape[4], srcTwiddleInAddr);
    auto srcMaskInt32Tile = DefineTile<int32_t>(1, srcShape[4], srcMaskAddr);
    auto srcMaskInt32MaxTile = DefineTile<int32_t>(1, AlignUp(srcTileW, 128), srcMaskAddr);
    auto srcMaskInt32KTile = DefineTile<int32_t>(1, k, srcMaskAddr);
    auto high1Tile = DefineTile<uint8_t, 32, false, 1>(1, 1, high1Addr);
    auto high2Tile = DefineTile<uint8_t, 32, false, 2>(2, 1, high1Addr);
    auto high3Tile = DefineTile<uint8_t, 32, false, 3>(3, 1, high1Addr);
    auto high1IntTile = DefineTile<int32_t>(1, 8, high1Addr);
    auto high2IntTile = DefineTile<int32_t>(1, 8, high2Addr);
    auto high3IntTile = DefineTile<int32_t>(1, 8, high3Addr);
    auto histogramUInt32Tile = DefineTile<uint32_t>(1, 256, histogramAddr);
    auto histogramInt32Tile = DefineTile<int32_t>(1, 256, histogramAddr);
    auto histogramTmpUInt32Tile = DefineTile<uint32_t>(1, 256, histogramTmpAddr);
    auto histogramTmpInt32Tile = DefineTile<int32_t>(1, 256, histogramTmpAddr);
    auto remindKTile = DefineTile<int32_t>(1, 8, remindKAddr);
    auto kthValueIntTile = DefineTile<int32_t>(1, 8, kthValueAddr);
    auto kthValueUInt32Tile = DefineTile<uint32_t>(1, 8, kthValueAddr);
    auto tmpIntTile = DefineTile<int32_t>(1, 8, tmpAddr);
    auto sortTempUInt32KTile = DefineTile<uint32_t>(1, k, sortTmpAddr);
    PTO_RS_SORT_TILE_DEFINE(32);

    for (LoopVar n0Index = 0; n0Index < srcShape[0]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < srcShape[1]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < srcShape[2]; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < srcShape[3]; ++n3Index) {
                    uint64_t srcAddr = src.GetAddr() + (n0Index * srcStride[0] + n1Index * srcStride[1] +
                                                        n2Index * srcStride[2] + n3Index * srcStride[3]) *
                                                           srcTypeSize;
                    uint64_t valAddr = value.GetAddr() + (n0Index * valStride[0] + n1Index * valStride[1] +
                                                          n2Index * valStride[2] + n3Index * valStride[3]) *
                                                             srcTypeSize;
                    uint64_t idxAddr = index.GetAddr() + (n0Index * idxStride[0] + n1Index * idxStride[1] +
                                                          n2Index * idxStride[2] + n3Index * idxStride[3]) *
                                                             idxTypeSize;
                    pto::TASSIGN(srcIntTile, srcAddr);
                    pto::TASSIGN(valIntTile, valAddr);
                    pto::TASSIGN(idxTile, idxAddr);
                    RadixSelectTwiddle<isLargest, true, isUInt, isFloat, SrcDType>(
                        twiddleIntTile, srcIntTile, srcMaskInt32Tile, cmpTile, uselessTile);
                    pto::TEXPANDS(kthValueIntTile, static_cast<int32_t>(0));
                    pto::TEXPANDS(high1Tile, static_cast<uint8_t>(0));
                    pto::THISTOGRAM<pto::HistByte::BYTE_3>(histogramUInt32Tile, twiddleUIntTile, high1Tile);
                    pto::TEXPANDS(remindKTile, static_cast<int32_t>(srcShape[4] - k));
                    pto::TCMPS(cmpTile, histogramInt32Tile, remindKTile, pto::CmpMode::GT);
                    pto::TCI<RSTile<uint32_t>::U, RSTile<uint32_t>::U, uint32_t, 0>(
                        histogramTmpUInt32Tile, static_cast<uint32_t>(0), histogramTmpUInt32Tile);
                    pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                    pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                    pto::TEXPANDS(gatherIntTile, 0);
                    pto::TGATHER(high1IntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TSHLS(tmpIntTile, high1IntTile, 24);
                    pto::TOR(kthValueIntTile, kthValueIntTile, tmpIntTile);
                    pto::TCMPS(cmpTile, high1IntTile, 0, pto::CmpMode::NE);
                    pto::TSUBS(rowMinTile, rowMinTile, 1);
                    pto::TSELS(rowMinTile, cmpTile, rowMinTile, uselessTile, 0);
                    pto::TGATHER(gatherIntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TGATHER(gatherIntTile, histogramInt32Tile, gatherIntTile, uselessTile);
                    pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                    pto::TSUB(remindKTile, remindKTile, gatherIntTile);
                    pto::THISTOGRAM<pto::HistByte::BYTE_2>(histogramUInt32Tile, twiddleUIntTile, high1Tile);
                    pto::TCMPS(cmpTile, histogramInt32Tile, remindKTile, pto::CmpMode::GT);
                    pto::TCI<RSTile<uint32_t>::U, RSTile<uint32_t>::U, uint32_t, 0>(
                        histogramTmpUInt32Tile, static_cast<uint32_t>(0), histogramTmpUInt32Tile);
                    pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                    pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                    pto::TEXPANDS(gatherIntTile, 0);
                    pto::TGATHER(high2IntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TSHLS(tmpIntTile, high2IntTile, 16);
                    pto::TOR(kthValueIntTile, kthValueIntTile, tmpIntTile);
                    pto::TCMPS(cmpTile, high2IntTile, 0, pto::CmpMode::NE);
                    pto::TSUBS(rowMinTile, rowMinTile, 1);
                    pto::TSELS(rowMinTile, cmpTile, rowMinTile, uselessTile, 0);
                    pto::TGATHER(gatherIntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TGATHER(gatherIntTile, histogramInt32Tile, gatherIntTile, uselessTile);
                    pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                    pto::TSUB(remindKTile, remindKTile, gatherIntTile);
                    pto::THISTOGRAM<pto::HistByte::BYTE_1>(histogramUInt32Tile, twiddleUIntTile, high2Tile);
                    pto::TCMPS(cmpTile, histogramInt32Tile, remindKTile, pto::CmpMode::GT);
                    pto::TCI<RSTile<uint32_t>::U, RSTile<uint32_t>::U, uint32_t, 0>(
                        histogramTmpUInt32Tile, static_cast<uint32_t>(0), histogramTmpUInt32Tile);
                    pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                    pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                    pto::TEXPANDS(gatherIntTile, 0);
                    pto::TGATHER(high3IntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TSHLS(tmpIntTile, high3IntTile, 8);
                    pto::TOR(kthValueIntTile, kthValueIntTile, tmpIntTile);
                    pto::TCMPS(cmpTile, high3IntTile, 0, pto::CmpMode::NE);
                    pto::TSUBS(rowMinTile, rowMinTile, 1);
                    pto::TSELS(rowMinTile, cmpTile, rowMinTile, uselessTile, 0);
                    pto::TGATHER(gatherIntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TGATHER(gatherIntTile, histogramInt32Tile, gatherIntTile, uselessTile);
                    pto::TSELS(gatherIntTile, cmpTile, gatherIntTile, uselessTile, 0);
                    pto::TSUB(remindKTile, remindKTile, gatherIntTile);
                    pto::THISTOGRAM<pto::HistByte::BYTE_0>(histogramUInt32Tile, twiddleUIntTile, high3Tile);
                    pto::TCMPS(cmpTile, histogramInt32Tile, remindKTile, pto::CmpMode::GT);
                    pto::TCI<RSTile<uint32_t>::U, RSTile<uint32_t>::U, uint32_t, 0>(
                        histogramTmpUInt32Tile, static_cast<uint32_t>(0), histogramTmpUInt32Tile);
                    pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                    pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                    pto::TEXPANDS(gatherIntTile, 0);
                    pto::TGATHER(high2IntTile, rowMinTile, gatherIntTile, uselessTile);
                    pto::TOR(kthValueIntTile, kthValueIntTile, high2IntTile);
                    pto::TEXPANDS(srcMaskInt32MaxTile, static_cast<int32_t>(0));
                    pto::TMOV(srcMaskInt32Tile, twiddleIntTile);
                    RadixSelectGather<pto::CmpMode::GT>(selectInt32GTTile, twiddleUIntTile, kthValueUInt32Tile,
                                                        selectCountUInt32GTTile, uselessTile);
                    RadixSelectGather<pto::CmpMode::EQ>(selectInt32EQTile, twiddleUIntTile, kthValueUInt32Tile,
                                                        selectCountUInt32EQTile, uselessTile);
                    pto::TCONCAT(idxTile, selectInt32GTTile, selectInt32EQTile, selectCountUInt32GTTile,
                                 selectCountUInt32EQTile);
                    pto::TEXPANDS(srcMaskInt32MaxTile, static_cast<int32_t>(0));
                    pto::TGATHER(srcMaskInt32KTile, twiddleIntTile, idxTile, uselessTile);
                    RadixSelectSortPrepare(sortTempInt32MaxTile, number0UInt32Tile, number1UInt32Tile,
                                           number2UInt32Tile, number3UInt32Tile);
                    RadixSelectSortTwoBit<0, 32>(
                        srcMaskInt32KTile, indexInt32Tile, sortTempInt32KTile, sortTempUInt32KTile, uselessTile,
                        select1Int32Tile, select2Int32Tile, select3Int32Tile, cnt1UInt32Tile, cnt2UInt32Tile,
                        cnt3UInt32Tile, number0UInt32Tile, number1UInt32Tile, number2UInt32Tile, number3UInt32Tile);
                    pto::TGATHER(select1Int32Tile, idxTile, indexInt32Tile, uselessTile);
                    pto::TMOV(idxTile, select1Int32Tile);
                    RadixSelectTwiddle<isLargest, false, isUInt, isFloat, SrcDType>(
                        valIntTile, srcMaskInt32KTile, twiddleIntKTile, cmpTile, uselessTile);
                }
            }
        }
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_B4__H
