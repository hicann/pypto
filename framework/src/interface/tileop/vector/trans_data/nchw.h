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
 * \file nchw.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TRANS_DATA_NCHW_H
#define TILEOP_TILE_OPERATOR_TRANS_DATA_NCHW_H
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_TRANSDATA_NCHW2NC1HWC0 TTransDataNCHW2NC1HWC0
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2NC1HWC0(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    if (inputN == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    Sync2_VS();
    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileC, tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, tileH * tileW);
        for (LoopVar i = 0; i < inputN; i++) {
            pto::TASSIGN(tmpInputTile,
                         (uint64_t)(input.GetAddr() + (i * tileC + inputC) * tileH * tileW * inputTypeSize));
            pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        }
        pipe_barrier(PIPE_V);
    }

    constexpr int elementSize = tileN * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<tileN, tileC, tileH, tileW>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                         pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAreaAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NCHW2Fractal_Z TTransDataNCHW2Fractal_Z
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2Fractal_Z(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;
    constexpr int elementSize = tileN * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;
    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (inputN == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<tileN, tileC, tileH, tileW>>;
    using tmpDst1TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                          pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmp1TileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                   tileH * tileW, C0>;
    inputTileData convInput;
    tmpDst1TileData convTmpDstNC1HWC0;
    tmp1TileData tmpTile;

    auto tmpDstNC1HWC0Addr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaTileAddr = tmpDstNC1HWC0Addr + elementSize;
    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDstNC1HWC0, (uint64_t)tmpDstNC1HWC0Addr);
    pto::TASSIGN(tmpTile, (uint64_t)tmpAreaTileAddr);

    auto inputPadN = (inputN + N0 - 1) / N0 * N0;
    auto padNSize = inputPadN - inputN;
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    Sync2_VS();
    if (padNSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileN, tileC * tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padNSize, tileC * tileH * tileW);
        pto::TASSIGN(tmpInputTile, (uint64_t)(input.GetAddr() + inputN * tileC * tileH * tileW * inputTypeSize));
        pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        pipe_barrier(PIPE_V);
    }

    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileC, tileH * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, tileH * tileW);
        for (LoopVar i = 0; i < inputN; i++) {
            pto::TASSIGN(tmpInputTile,
                         (uint64_t)(input.GetAddr() + (i * tileC + inputC) * tileH * tileW * inputTypeSize));
            pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        }
        pipe_barrier(PIPE_V);
    }

    pto::TTRANS(convTmpDstNC1HWC0, convInput, tmpTile);
    pipe_barrier(PIPE_V);
    constexpr int64_t tileN1 = tileN / N0;
    using tmpDst2TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                          pto::ConvTileShape<tileC1 * tileH * tileW, tileN1, N0, C0>>;
    tmpDst2TileData dstFractalZ;
    pto::TASSIGN(dstFractalZ, (uint64_t)dst.GetAddr());
    pto::TTRANS(dstFractalZ, convTmpDstNC1HWC0, tmpTile);
}

#define OP_TILE_OP_TRANSDATA_NC1HWC02NCHW TTransDataNC1HWC02NCHW
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNC1HWC02NCHW(DST dst, TMP tmpTensor, INPUT input)
{
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto C0 = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (inputN == 0 || inputC1 == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    constexpr int elementSize = tileN * tileC1 * tileH * tileW * C0;
    constexpr int bufferSize = elementSize * inputTypeSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<tileN, tileC1 * C0, tileH, tileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpTensor.GetAddr());
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_FractalZ2NCHW TTransDataFractalZ2NCHW
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataFractalZ2NCHW(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto TileC1HW = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto TileN1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;
    constexpr auto TileC1 = TileC1HW / dstTileH / dstTileW;
    constexpr int elementSize = TileC1HW * TileN1 * N0 * C0;
    constexpr int bufferSize = elementSize * inputTypeSize;
    constexpr int TileN = N0 * TileN1;

    const auto inputLayout = input.GetLayout();
    auto inputC1HW = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputN1 = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputN0 = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using Tdst = typename INPUT::Type;
    using Tsrc = typename INPUT::Type;
    using Ttmp = typename INPUT::Type;

    __ubuf__ Tsrc* inputAddr = (__ubuf__ Tsrc*)((uint64_t)(input.GetAddr()));
    __ubuf__ Tdst* dstAddr = (__ubuf__ Tdst*)((uint64_t)(dst.GetAddr()));
    __ubuf__ Ttmp* tmpAddr = (__ubuf__ Ttmp*)((uint64_t)(tmpTensor.GetAddr()));

    if (inputC1HW == 0 || inputN1 == 0 || inputN0 == 0 || inputC0 == 0) {
        return;
    }

    uint32_t burstNum = TileN;
    uint32_t lenBurst = (C0 * inputTypeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE;
    uint32_t srcGap = 0;
    uint32_t dstGap = (TileC1HW * C0 * inputTypeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE - lenBurst;
    for (int i = 0; i < TileC1HW; i++) {
        __ubuf__ Tsrc* srcPtr = inputAddr + i * TileN * C0;
        __ubuf__ Tdst* tmpPtr = tmpAddr + i * C0;
        pto::pto_copy_ubuf_to_ubuf(tmpPtr, srcPtr, burstNum, lenBurst, srcGap, dstGap);
    }
    pipe_barrier(PIPE_V);

    __ubuf__ Ttmp* tmpAreaTileAddr = tmpAddr + elementSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<TileN, TileC1, dstTileH, dstTileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<TileN, TileC1 * C0, dstTileH, dstTileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, dstTileH * dstTileW, C0,
                                  pto::BLayout::RowMajor, dstTileH * dstTileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)(tmpAddr));
    pto::TASSIGN(convTmpDst, (uint64_t)(dstAddr));
    pto::TASSIGN(tmpAreaTile, (uint64_t)(tmpAreaTileAddr));
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#endif
