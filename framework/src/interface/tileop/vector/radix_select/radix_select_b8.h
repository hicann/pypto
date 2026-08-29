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
 * \file radix_select_b8.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_B8__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_B8__H
#if defined(PTO_NPU_ARCH_A5)

#include "radix_select_util_b8.h"

namespace RadixSelectUtil {
/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint64  |
hisSrcTmpAddr     | srcMaskShape    uint64  |
hisSrcAddr        | srcMaskShape    uint16  |
cmpAddr           | cmpAlign        uint8   |
maskAddr          | cmpAlign        uint8   |
highAddr          | 32              uint8   |
tmpAddr           | 4               uint64  |
remaindKAddr      | 8               uint32  |
kthValueAddr      | 4               uint64  |
selectCountGTAddr | 8               uint32  | and rowMinAddr and tciTmp1Addr
selectCountEQAddr | 8               uint32  | and gatherAddr
uselessAddr       | 8               uint32  |
histogramAddr     | 256             uint32  | selectGTAddr  | kAlign uint32 |
histogramTmpAddr  | 256             uint32  | selectEQAddr  | kAlign uint32 |
tciAddr           | 256             uint32  |
*/
template <int k, bool isLargest, typename VAL, typename IDX, typename TMP, typename SRC>
TILEOP void RadixSelectCalcB8(VAL value, IDX index, TMP tmp, SRC src)
{
    PTO_RS_PREPARE;
    static_assert(std::is_same_v<SrcDType, uint64_t> || std::is_same_v<SrcDType, int64_t>);
    constexpr bool isUInt = std::is_same_v<SrcDType, uint64_t>;
    constexpr int64_t srcTileWB2 = AlignUp(srcTileW, 16);
    size_t row = srcShape[3];
    // Define memory address
    size_t point = tmp.GetAddr();
    size_t srcMaskAddr = DefineWorkSpace<uint64_t, srcTileH * srcMaskShape>(point);
    size_t hisSrcTmpAddr = DefineWorkSpace<uint64_t, srcTileH * srcMaskShape>(point);
    size_t hisSrcAddr = DefineWorkSpace<uint16_t, srcTileH * srcMaskShape>(point);
    size_t cmpAddr = DefineWorkSpace<uint8_t, srcTileH * cmpAlign>(point);
    size_t maskAddr = DefineWorkSpace<uint8_t, srcTileH * cmpAlign>(point);
    size_t highAddr = DefineWorkSpace<uint8_t, srcTileH * 32>(point);
    size_t tmpAddr = DefineWorkSpace<uint64_t, srcTileH * 4>(point);
    size_t remaindKAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t kthValueAddr = DefineWorkSpace<uint64_t, srcTileH * 4>(point);
    size_t selectCountGTAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t rowMinAddr = selectCountGTAddr;
    size_t tciTmp1Addr = selectCountGTAddr;
    size_t selectCountEQAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t gatherAddr = selectCountEQAddr;
    size_t uselessAddr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t histogramAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t histogramTmpAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    size_t tciAddr = DefineWorkSpace<uint32_t, srcTileH * 256>(point);
    point = histogramAddr;
    size_t selectGTAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    size_t selectEQAddr = DefineWorkSpace<uint32_t, srcTileH * kAlign>(point);
    point = hisSrcTmpAddr;
    size_t sortTmpAddr = DefineWorkSpace<uint64_t, srcTileH * srcMaskShape>(point);
    size_t number0Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);
    size_t number1Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);
    size_t number2Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);
    size_t number3Addr = DefineWorkSpace<uint16_t, srcTileH * 16>(point);
    size_t cnt1Addr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t cnt2Addr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t cnt3Addr = DefineWorkSpace<uint32_t, srcTileH * 8>(point);
    size_t select1Addr = DefineWorkSpace<uint32_t, srcTileH * srcMaskShape>(point);
    size_t select2Addr = DefineWorkSpace<uint32_t, srcTileH * srcMaskShape>(point);
    size_t sortTmp2Addr = select2Addr;
    size_t select3Addr = DefineWorkSpace<uint32_t, srcTileH * srcMaskShape>(point);
    size_t indexAddr = DefineWorkSpace<uint32_t, srcTileH * srcMaskShape>(point);
    // Define tile
    auto srcIntTile = DefineTile<ConvIntType, srcTileH, srcTileW>(row, srcShape[4]);
    auto valIntTile = DefineTile<ConvIntType, srcTileH, valTileW>(row, k);
    auto idxTile = DefineTile<IdxDType, srcTileH, idxTileW>(row, k);
    auto cmpTile = DefineTile<uint8_t, srcTileH, cmpAlign>(row, cmpSize, cmpAddr);
    auto selectInt32GTTile = DefineTile<int32_t, srcTileH, kAlign>(row, k, selectGTAddr);
    auto selectInt32EQTile = DefineTile<int32_t, srcTileH, kAlign>(row, k, selectEQAddr);
    auto selectCountUInt32GTTile = DefineTile<uint32_t, srcTileH>(row, 1, selectCountGTAddr);
    auto selectCountUInt32EQTile = DefineTile<uint32_t, srcTileH>(row, 1, selectCountEQAddr);
    auto rowMinTile = DefineTile<int32_t, srcTileH>(row, 8, rowMinAddr);
    auto tciTmp1Tile = DefineTile<uint32_t, srcTileH>(row, 8, tciTmp1Addr);
    auto gatherIntTile = DefineTile<int32_t, srcTileH>(row, 8, gatherAddr);
    auto uselessTile = DefineTile<uint32_t>(1, 1, uselessAddr);
    auto maskTile = DefineTile<uint8_t, srcTileH, cmpAlign>(row, cmpSize, maskAddr);
    auto hisSrcTmpUInt64Tile = DefineTile<uint64_t, srcTileH, srcMaskShape>(row, srcMaskShape, hisSrcTmpAddr);
    auto hisSrcTmpInt64Tile = DefineTile<int64_t, srcTileH, srcMaskShape>(row, srcMaskShape, hisSrcTmpAddr);
    auto hisSrcTmpInt64MinTile = DefineTile<int64_t, srcTileH, srcMaskShape>(row, srcShape[4], hisSrcTmpAddr);
    auto hisSrcTmpUInt16Tile = DefineTile<uint16_t, srcTileH, srcMaskShape>(row, srcMaskShape, hisSrcTmpAddr);
    auto hisSrcUInt16Tile = DefineTile<uint16_t, srcTileH, srcMaskShape>(row, srcMaskShape, hisSrcAddr);
    auto hisSrcInt16Tile = DefineTile<int16_t, srcTileH, srcMaskShape>(row, srcMaskShape, hisSrcAddr);
    auto srcMaskUInt64Tile = DefineTile<uint64_t, srcTileH, srcMaskShape>(row, srcMaskShape, srcMaskAddr);
    auto srcMaskUInt64KTile = DefineTile<uint64_t, srcTileH, srcMaskShape>(row, k, srcMaskAddr);
    auto srcMaskInt64MinTile = DefineTile<int64_t, srcTileH, srcMaskShape>(row, srcShape[4], srcMaskAddr);
    auto srcMaskInt64Tile = DefineTile<int64_t, srcTileH, srcMaskShape>(row, srcMaskShape, srcMaskAddr);
    auto srcMaskInt64KTile = DefineTile<int64_t, srcTileH, srcMaskShape>(row, k, srcMaskAddr);
    auto srcMaskInt32FullTile = DefineTile<int32_t, srcTileH, srcMaskShape * 2>(row, srcMaskShape * 2, srcMaskAddr);
    auto srcMaskUInt16FullTile = DefineTile<uint16_t, srcTileH, srcMaskShape * 4>(row, srcMaskShape * 4, srcMaskAddr);
    auto srcMaskInt16Tile = DefineTile<int16_t, srcTileH, srcMaskShape>(row, srcMaskShape, srcMaskAddr);
    auto srcMaskInt16KTile = DefineTile<int16_t, srcTileH, srcMaskShape>(row, k, srcMaskAddr);
    auto tmpUInt32Tile = DefineTile<uint32_t, srcTileH>(row, 8, tmpAddr);
    auto tmpUInt16Tile = DefineTile<uint16_t, srcTileH>(row, 16, tmpAddr);
    auto remaindKInt32Tile = DefineTile<int32_t, srcTileH>(row, 8, remaindKAddr);
    auto kUInt64Tile = DefineTile<uint64_t, srcTileH>(row, 4, kthValueAddr);
    auto kUInt32Tile = DefineTile<uint32_t, srcTileH>(row, 8, kthValueAddr);
    auto highTile = DefineTile<uint8_t, srcTileH, 32, true>(1, row, highAddr);
    auto highIntTile = DefineTile<int32_t, srcTileH>(row, 8, highAddr);
    auto highUInt32Tile = DefineTile<uint32_t, srcTileH>(row, 8, highAddr);
    auto highOneColUInt16Tile = DefineTile<uint16_t, srcTileH>(row, 1, highAddr);
    auto histogramUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(row, 256, histogramAddr);
    auto histogramInt32Tile = DefineTile<int32_t, srcTileH, 256>(row, 256, histogramAddr);
    auto histogramInt32PreTile = DefineTile<int32_t, srcTileH, 256>(row, 256, histogramAddr - 32);
    auto histogramTmpUInt32Tile = DefineTile<uint32_t, srcTileH, 256>(row, 256, histogramTmpAddr);
    auto histogramTmpInt32Tile = DefineTile<int32_t, srcTileH, 256>(row, 256, histogramTmpAddr);
    auto tciTile = DefineTile<uint32_t, srcTileH, 256>(row, 256, tciAddr);
    auto number0UInt16Tile = DefineTile<uint16_t>(row, 16, number0Addr);
    auto number1UInt16Tile = DefineTile<uint16_t>(row, 16, number1Addr);
    auto number2UInt16Tile = DefineTile<uint16_t>(row, 16, number2Addr);
    auto number3UInt16Tile = DefineTile<uint16_t>(row, 16, number3Addr);
    auto cnt1UInt32Tile = DefineTile<uint32_t, srcTileH>(row, 8, cnt1Addr);
    auto cnt2UInt32Tile = DefineTile<uint32_t, srcTileH>(row, 8, cnt2Addr);
    auto cnt3UInt32Tile = DefineTile<uint32_t, srcTileH>(row, 8, cnt3Addr);
    auto select1Int32Tile = DefineTile<int32_t, srcTileH, srcMaskShape>(row, k, select1Addr);
    auto select2Int32Tile = DefineTile<int32_t, srcTileH, srcMaskShape>(row, k, select2Addr);
    auto select3Int32Tile = DefineTile<int32_t, srcTileH, srcMaskShape>(row, k, select3Addr);
    auto indexInt32Tile = DefineTile<int32_t, srcTileH, srcMaskShape>(row, k, indexAddr);
    auto sortTempUInt64KTile = DefineTile<uint64_t, srcTileH, srcMaskShape>(row, k, sortTmpAddr);
    auto sortTempInt32Tile = DefineTile<int32_t, srcTileH, srcMaskShape * 2>(row, srcMaskShape * 2, sortTmpAddr);
    auto sortTempUInt16FullTile = DefineTile<uint16_t, srcTileH, srcMaskShape * 4>(row, srcMaskShape * 4, sortTmpAddr);
    auto sortTempInt32FullTile = DefineTile<int32_t, srcTileH, srcMaskShape * 2>(row, srcMaskShape * 2, sortTmpAddr);
    auto sortTempInt16Tile = DefineTile<int16_t, srcTileH, srcMaskShape>(row, srcMaskShape, sortTmpAddr);
    auto sortTempInt16KTile = DefineTile<int16_t, srcTileH, srcMaskShape>(row, k, sortTmpAddr);
    auto sortTemp2UInt16Tile = DefineTile<uint16_t, srcTileH, srcMaskShape>(row, srcMaskShape, sortTmp2Addr);
    auto sortTemp2UInt16KTile = DefineTile<uint16_t, srcTileH, srcMaskShape>(row, k, sortTmp2Addr);

    for (LoopVar n0Index = 0; n0Index < srcShape[0]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < srcShape[1]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < srcShape[2]; ++n2Index) {
                RadixSelectBatchAssign(src, value, index, n0Index, n1Index, n2Index, srcStride, valStride, idxStride,
                                       srcTypeSize, idxTypeSize, srcIntTile, valIntTile, idxTile);
                RadixSelectTCI(tciTile);
                pto::TEXPANDS(srcMaskInt64Tile, 0);
                pto::TEXPANDS(maskTile, 0xff);
                pto::TEXPANDS(remaindKInt32Tile, static_cast<int32_t>(srcMaskShape - k));
                pto::TEXPANDS(highIntTile, 0);
                pto::TEXPANDS(kUInt32Tile, 0);
                RadixSelectTwiddleB8<isLargest, true, isUInt, SrcDType>(srcMaskInt64MinTile, srcIntTile,
                                                                        hisSrcTmpInt64MinTile);
                // 0-8bit
                RadixSelectHistogramPrepareB8<0>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile, maskTile,
                                                 uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<0>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 8-16bit
                RadixSelectHistogramPrepareB8<8>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile, maskTile,
                                                 uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<8>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 16-24bit
                RadixSelectHistogramPrepareB8<16>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile,
                                                  maskTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<16>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 24-32bit
                RadixSelectHistogramPrepareB8<24>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile,
                                                  maskTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<24>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 32-40bit
                RadixSelectHistogramPrepareB8<32>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile,
                                                  maskTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<32>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 40-48bit
                RadixSelectHistogramPrepareB8<40>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile,
                                                  maskTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<40>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 48-56bit
                RadixSelectHistogramPrepareB8<48>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile,
                                                  maskTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<48>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                RadixSelectUpdateMaskB8(hisSrcUInt16Tile, highOneColUInt16Tile, maskTile, cmpTile);
                RadixSelectUpdateRemaindKB8(remaindKInt32Tile, histogramInt32PreTile, highIntTile, gatherIntTile,
                                            rowMinTile, cmpTile, uselessTile);
                // 56-64bit
                RadixSelectHistogramPrepareB8<56>(hisSrcUInt16Tile, hisSrcTmpUInt16Tile, srcMaskUInt16FullTile,
                                                  maskTile, uselessTile);
                RadixSelectHistogram<pto::HistByte::BYTE_0>(histogramUInt32Tile, hisSrcUInt16Tile, highTile);
                RadixSelectCalcKTHBitB8(histogramInt32Tile, histogramTmpInt32Tile, histogramTmpUInt32Tile, highIntTile,
                                        remaindKInt32Tile, cmpTile, tciTile, uselessTile);
                RadixSelectUpdateKTHValueB8<56>(kUInt32Tile, highUInt32Tile, tciTmp1Tile, tmpUInt32Tile);
                pto::TROWEXPAND(hisSrcTmpUInt64Tile, kUInt64Tile);
                pto::TEXPANDS(hisSrcUInt16Tile, 0);
                RadixSelectFinalSelectB8(selectInt32GTTile, selectInt32EQTile, srcMaskUInt64Tile, hisSrcTmpUInt64Tile,
                                         hisSrcInt16Tile, tmpUInt16Tile, selectCountUInt32GTTile,
                                         selectCountUInt32EQTile, idxTile, cmpTile, uselessTile);
                RadixSelectMoveB8(hisSrcTmpUInt64Tile, srcMaskUInt64Tile);
                pto::TEXPANDS(srcMaskInt32FullTile, 0x7fff);
                RadixSelectGather(srcMaskInt64KTile, hisSrcTmpInt64Tile, idxTile, uselessTile);
                RadixSelectSortPrepare(sortTempInt32Tile, number0UInt16Tile, number1UInt16Tile, number2UInt16Tile,
                                       number3UInt16Tile);
                RadixSelectSortTwoBitB8<0, 64>(
                    srcMaskUInt16FullTile, srcMaskUInt64KTile, srcMaskInt32FullTile, srcMaskInt16Tile,
                    srcMaskInt16KTile, indexInt32Tile, sortTempUInt16FullTile, sortTempUInt64KTile,
                    sortTempInt32FullTile, sortTempInt16Tile, sortTempInt16KTile, sortTemp2UInt16Tile,
                    sortTemp2UInt16KTile, uselessTile, select1Int32Tile, select2Int32Tile, select3Int32Tile,
                    cnt1UInt32Tile, cnt2UInt32Tile, cnt3UInt32Tile, number0UInt16Tile, number1UInt16Tile,
                    number2UInt16Tile, number3UInt16Tile);
                RadixSelectGather(select1Int32Tile, idxTile, indexInt32Tile, uselessTile);
                pto::TMOV(idxTile, select1Int32Tile);
                RadixSelectTwiddleB8<isLargest, false, isUInt, SrcDType>(valIntTile, srcMaskInt64KTile,
                                                                         hisSrcTmpInt64Tile);
            }
        }
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_B8__H
