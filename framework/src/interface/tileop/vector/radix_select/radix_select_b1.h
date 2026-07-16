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
 * \file radix_select_b1.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_B1__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_B1__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_util.h"

namespace RadixSelectUtil {
/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint16  |
srcTwiddleInAddr  | srcTileW        srcType |
cmpAddr           | cmpAlign        uint8   | srcTmpAddr       | srcMaskShape uint16  |
highAddr          | 32              uint8   |
selectCountGTAddr | 8               uint32  | and rowMinAddr
selectCountEQAddr | 8               uint32  | and gatherAddr
uselessAddr       | 8               uint32  |
histogramAddr     | 256             uint32  | selectGTAddr     | kAlign uint32 |
histogramTmpAddr  | 256             uint32  | selectEQAddr     | kAlign uint32 |
*/
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalcB1(VAL value, IDX index, TMP tmp, SRC src)
{
    PTO_RS_PREPARE;
    constexpr bool isUInt = std::is_same_v<SrcDType, uint8_t>;
    constexpr bool isFloat = !(std::is_same_v<SrcDType, uint8_t> || std::is_same_v<SrcDType, int8_t>);
    // Define memory address
    size_t point = tmp.GetAddr();
    size_t srcMaskAddr = DefineWorkSpace<uint16_t>(point, AlignUp(srcTileW, 128));
    size_t sortTmpAddr_ = point;
    size_t srcTwiddleInAddr = DefineWorkSpace<SrcDType>(point, srcTileW);
    size_t cmpAddr = DefineWorkSpace<uint8_t>(point, AlignUp(cmpSize, 32));
    size_t srcTmpAddr = cmpAddr;
    size_t highAddr = DefineWorkSpace<uint8_t>(point, 32);
    size_t selectCountGTAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t rowMinAddr = selectCountGTAddr;
    size_t selectCountEQAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t gatherAddr = selectCountEQAddr;
    size_t uselessAddr = DefineWorkSpace<uint32_t>(point, 8);
    size_t histogramAddr = DefineWorkSpace<uint32_t>(point, 256);
    size_t histogramTmpAddr = DefineWorkSpace<uint32_t>(point, 256);
    point = histogramAddr;
    size_t selectGTAddr = DefineWorkSpace<uint32_t>(point, kAlign);
    size_t selectEQAddr = DefineWorkSpace<uint32_t>(point, kAlign);
    point = sortTmpAddr_;
    size_t sortTmpAddr = DefineWorkSpace<uint16_t>(point, kAlign);
    PTO_RS_SORT_ADDR_DEFINE;
    // Define tile
    PTO_RS_COMMON_TILE_DEFINE;
    auto twiddleIntTile = DefineTile<ConvIntType>(1, srcShape[4], srcTwiddleInAddr);
    auto twiddleUIntTile = DefineTile<ConvUIntType>(1, srcShape[4], srcTwiddleInAddr);
    auto srcMaskUInt8KTile = DefineTile<uint8_t>(1, k, srcMaskAddr);
    auto srcMaskInt8KTile = DefineTile<int8_t>(1, k, srcMaskAddr);
    auto srcMaskUInt16Tile = DefineTile<uint16_t>(1, srcShape[4], srcMaskAddr);
    auto srcMaskInt16Tile = DefineTile<int16_t>(1, srcShape[4], srcMaskAddr);
    auto srcMaskInt16KTile = DefineTile<int16_t>(1, k, srcMaskAddr);
    auto srcMaskInt16MaxTile = DefineTile<int16_t>(1, AlignUp(srcTileW, 128), srcMaskAddr);
    auto srcTmpInt16KTile = DefineTile<int16_t>(1, k, srcTmpAddr);
    auto highTile = DefineTile<uint8_t, 32, true>(1, 1, highAddr);
    auto highIntTile = DefineTile<int32_t>(1, 8, highAddr);
    auto highUInt16Tile = DefineTile<uint16_t>(1, 16, highAddr);
    auto histogramUInt32Tile = DefineTile<uint32_t>(1, 256, histogramAddr);
    auto histogramTmpUInt32Tile = DefineTile<uint32_t>(1, 256, histogramTmpAddr);
    auto histogramTmpInt32Tile = DefineTile<int32_t>(1, 256, histogramTmpAddr);
    PTO_RS_SORT_TILE_DEFINE(16);

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
                        twiddleIntTile, srcIntTile, uselessTile, uselessTile, uselessTile);
                    pto::TEXPANDS(srcMaskInt16MaxTile, static_cast<int16_t>(0));
                    pto::TCVT(srcMaskUInt16Tile, twiddleUIntTile, pto::RoundMode::CAST_TRUNC);
                    pto::TEXPANDS(highTile, static_cast<uint8_t>(0));
                    pto::THISTOGRAM<pto::HistByte::BYTE_0>(histogramUInt32Tile, srcMaskUInt16Tile, highTile);
                    pto::TCMPS(cmpTile, histogramUInt32Tile, static_cast<uint32_t>(srcShape[4] - k), pto::CmpMode::GT);
                    pto::TCI<RSTile<uint32_t>::U, RSTile<uint32_t>::U, uint32_t, 0>(
                        histogramTmpUInt32Tile, static_cast<uint32_t>(0), histogramTmpUInt32Tile);
                    pto::TSELS(histogramTmpUInt32Tile, cmpTile, histogramTmpUInt32Tile, uselessTile, 0x7fffffffu);
                    pto::TROWMIN(rowMinTile, histogramTmpInt32Tile, uselessTile);
                    pto::TEXPANDS(gatherIntTile, 0);
                    pto::TGATHER(highIntTile, rowMinTile, gatherIntTile, uselessTile);
                    RadixSelectGather<pto::CmpMode::GT>(selectInt32GTTile, srcMaskInt16Tile, highUInt16Tile,
                                                        selectCountUInt32GTTile, uselessTile);
                    RadixSelectGather<pto::CmpMode::EQ>(selectInt32EQTile, srcMaskInt16Tile, highUInt16Tile,
                                                        selectCountUInt32EQTile, uselessTile);
                    pto::TCONCAT(idxTile, selectInt32GTTile, selectInt32EQTile, selectCountUInt32GTTile,
                                 selectCountUInt32EQTile);
                    pto::TGATHER(srcTmpInt16KTile, srcMaskInt16Tile, idxTile, uselessTile);
                    pto::TEXPANDS(srcMaskInt16MaxTile, 0);
                    pto::TMOV(srcMaskInt16KTile, srcTmpInt16KTile);
                    RadixSelectSortPrepare(sortTempInt16MaxTile, number0UInt16Tile, number1UInt16Tile,
                                           number2UInt16Tile, number3UInt16Tile);
                    RadixSelectSortTwoBit<0, 8>(
                        srcMaskInt16KTile, indexInt32Tile, sortTempInt16KTile, sortTempInt16KTile, uselessTile,
                        select1Int32Tile, select2Int32Tile, select3Int32Tile, cnt1UInt32Tile, cnt2UInt32Tile,
                        cnt3UInt32Tile, number0UInt16Tile, number1UInt16Tile, number2UInt16Tile, number3UInt16Tile);
                    pto::TGATHER(select1Int32Tile, idxTile, indexInt32Tile, uselessTile);
                    pto::TMOV(idxTile, select1Int32Tile);
                    pto::TCVT(srcMaskUInt8KTile, srcMaskInt16KTile, pto::RoundMode::CAST_TRUNC);
                    RadixSelectTwiddle<isLargest, false, isUInt, isFloat, SrcDType>(
                        valIntTile, srcMaskInt8KTile, uselessTile, uselessTile, uselessTile);
                }
            }
        }
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_B1__H
