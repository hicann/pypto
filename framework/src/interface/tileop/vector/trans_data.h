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
 * \file trans_data.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TRANS_DATA__H
#define TILEOP_TILE_OPERATOR_TRANS_DATA__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

TILEOP void Sync23_VS()
{
#ifdef __DAV_V220
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#endif
}

TILEOP void Sync2_VS()
{
#ifdef __DAV_V220
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
#endif
}

TILEOP void SyncVS_3()
{
#ifdef __DAV_V220
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
#endif
}

#define OP_TILE_OP_TRANSDATA_NCHW2NC1HWC0 TTransDataNCHW2NC1HWC0
template <int N, int C, int H, int W, typename DST, typename TYPEC, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2NC1HWC0(DST dst, TYPEC coordinate, TMP tmpTensor, INPUT input, int n, int c,
                                              int h, int w)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;

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
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaAddr = tmpDstAddr + elementSize;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);

    Sync2_VS();
    pipe_barrier(PIPE_ALL);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
    SyncVS_3();
    pipe_barrier(PIPE_ALL);

    const auto inputLayout = input.GetLayout();
    const auto gmLayout = dst.GetLayout();

    constexpr auto tmpDstStride0 = tileC1 * tileH * tileW * C0;
    constexpr auto tmpDstStride1 = tileH * tileW * C0;
    constexpr auto tmpDstStride2 = tileW * C0;
    constexpr auto tmpDstStride3 = C0;

    const auto dstStride0 = gmLayout.template GetStrideDim<DIM_1ST, 5>();
    const auto dstStride1 = gmLayout.template GetStrideDim<DIM_2ND, 5>();
    const auto dstStride2 = gmLayout.template GetStrideDim<DIM_3RD, 5>();
    const auto dstStride3 = gmLayout.template GetStrideDim<DIM_4TH, 5>();

    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 5>(coordinate));
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputC1 = (inputC + C0 - 1) / C0;
    using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileW, C0, pto::BLayout::RowMajor, -1, -1>;
    using GlobalData = pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>,
                                         pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDstTile(inputW, C0);
    for (LoopVar i = 0; i < inputN; i++) {
        for (LoopVar j = 0; j < inputC1; j++) {
            for (LoopVar k = 0; k < inputH; k++) {
                uint64_t tmpDstStride = i * tmpDstStride0 + j * tmpDstStride1 + k * tmpDstStride2;
                uint64_t DstStride = (n + i) * dstStride0 + (c / C0 + j) * dstStride1 + (h + k) * dstStride2 +
                                     w * dstStride3;
                pto::TASSIGN(tmpDstTile, (uint64_t)(tmpDstAddr + tmpDstStride));
                GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, inputW, C0),
                                      pto::Stride(1, 1, 1, C0, 1));
                pto::TSTORE(globalData, tmpDstTile);
            }
        }
    }
}

#define OP_TILE_OP_TRANSDATA_NCHW2Fractal_Z TTransDataNCHW2Fractal_Z
template <int N, int C, int H, int W, typename DST, typename TYPEC, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2Fractal_Z(DST dst, TYPEC coordinate, TMP tmpTensor, INPUT input, int n, int c,
                                                int h, int w, int groupIndex, int group)
{
    n = n % N;
    c = c % C;
    h = h % H;
    w = w % W;
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;
    constexpr int elementSize = tileN * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

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
    auto tmpDstFractalZAddr = tmpDstNC1HWC0Addr + elementSize;
    auto tmpAreaTileAddr = tmpDstFractalZAddr + elementSize;
    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDstNC1HWC0, (uint64_t)tmpDstNC1HWC0Addr);
    pto::TASSIGN(tmpTile, (uint64_t)tmpAreaTileAddr);

    Sync2_VS();
    pipe_barrier(PIPE_ALL);
    pto::TTRANS(convTmpDstNC1HWC0, convInput, tmpTile);
    SyncV();
    pipe_barrier(PIPE_ALL);

    constexpr int64_t N0 = 16;
    constexpr int64_t tileN1 = tileN / N0;
    using tmpDst2TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                          pto::ConvTileShape<tileC1 * tileH * tileW, tileN1, N0, C0>>;
    tmpDst2TileData convTmpDstFractalZ;
    pto::TASSIGN(convTmpDstFractalZ, (uint64_t)tmpDstFractalZAddr);

    pipe_barrier(PIPE_ALL);
    pto::TTRANS(convTmpDstFractalZ, convTmpDstNC1HWC0, tmpTile);
    SyncVS_3();
    pipe_barrier(PIPE_ALL);

    const auto inputLayout = input.GetLayout();
    const auto gmLayout = dst.GetLayout();
    constexpr auto dstFractalZStride0 = tileN * C0;
    constexpr auto dstFractalZStride1 = N0 * C0;
    constexpr auto dstFractalZStride2 = C0;

    const auto dstStride0 = gmLayout.template GetStrideDim<DIM_1ST, 4>();
    const auto dstStride1 = gmLayout.template GetStrideDim<DIM_2ND, 4>();
    const auto dstStride2 = gmLayout.template GetStrideDim<DIM_3RD, 4>();

    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    DstAddr = DstAddr + groupIndex * N * C * H * W;
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 5>(coordinate));
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputC1 = (inputC + C0 - 1) / C0;
    auto inputN1 = (inputN + N0 - 1) / N0;
    using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, N0, C0, pto::BLayout::RowMajor, -1, -1>;
    using GlobalData = pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>,
                                         pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDst2Tile(N0, C0);

    int64_t offsetC1 = c / C0;
    int64_t offsetN1 = n / N0;

    for (LoopVar i = 0; i < inputC1; i++) {
        for (LoopVar k = 0; k < inputH; k++) {
            int64_t idx = (offsetC1 + i) * H * W + (h + k) * W + w;
            for (LoopVar m = 0; m < inputW; m++) {
                int64_t tmpDst2Idx = i * tileH * tileW + k * tileW + m;
                for (LoopVar j = 0; j < inputN1; j++) {
                    uint64_t tmpDst2Stride = tmpDst2Idx * dstFractalZStride0 + j * dstFractalZStride1;
                    uint64_t DstStride = idx * dstStride0 + (j + offsetN1) * dstStride1;
                    pto::TASSIGN(tmpDst2Tile, (uint64_t)(tmpDstFractalZAddr + tmpDst2Stride));
                    GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, N0, C0),
                                          pto::Stride(1, 1, 1, C0, 1));
                    pto::TSTORE(globalData, tmpDst2Tile);
                }
                idx++;
            }
        }
    }
}

#define OP_TILE_OP_TRANSDATA_NC1HWC02NCHW TTransDataNC1HWC02NCHW
template <int N, int dstC, int H, int W, typename DST, typename TYPEC, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNC1HWC02NCHW(DST dst, TYPEC coordinate, TMP tmpTensor, INPUT input, int n, int c1,
                                              int h, int w, int c0, int groupIndex, int group, int padSize)
{
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto C0 = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;

    auto perDstGroupC = dstC / group;
    auto perDstGroupC1 = (perDstGroupC + C0 - 1) / C0;
    auto dstCStart = groupIndex * perDstGroupC + (c1 % perDstGroupC1) * C0;

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
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaAddr = tmpDstAddr + elementSize;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);

    Sync2_VS();
    pipe_barrier(PIPE_ALL);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
    SyncVS_3();
    pipe_barrier(PIPE_ALL);

    const auto inputLayout = input.GetLayout();
    const auto gmLayout = dst.GetLayout();

    constexpr auto tmpDstStride0 = tileC1 * C0 * tileH * tileW;
    constexpr auto tmpDstStride1 = tileH * tileW;
    constexpr auto tmpDstStride2 = tileW;

    const auto dstStride0 = gmLayout.template GetStrideDim<DIM_1ST, 4>();
    const auto dstStride1 = gmLayout.template GetStrideDim<DIM_2ND, 4>();
    const auto dstStride2 = gmLayout.template GetStrideDim<DIM_3RD, 4>();

    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 5>(coordinate));
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto realTileW = (tileW + C0 - 1) / C0 * C0;
    using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, 1, realTileW, pto::BLayout::RowMajor, -1,
                                 -1>;
    using GlobalData = pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>,
                                         pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDstTile(1, inputW);
    auto cValidLen = inputC1 * C0 - padSize;

    // 处理尾部数据
    if (tileW % C0 != 0) {
        for (LoopVar i = 0; i < inputN; i++) {
            for (LoopVar j = 0; j < cValidLen; j++) {
                for (LoopVar k = 0; k < inputH; k++) {
                    uint64_t tmpDstStride = i * tmpDstStride0 + j * tmpDstStride1 + k * tmpDstStride2;
                    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                    for (LoopVar m = 0; m < inputW; m++) {
                        tmpAreaAddr[m] = tmpDstAddr[tmpDstStride + m];
                    }
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                    pto::TASSIGN(tmpDstTile, (uint64_t)(tmpAreaAddr));
                    uint64_t DstStride = (n + i) * dstStride0 + (dstCStart + j) * dstStride1 + (h + k) * dstStride2 + w;
                    GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, 1, inputW),
                                          pto::Stride(1, 1, 1, inputW, 1));
                    pto::TSTORE(globalData, tmpDstTile);
                }
            }
        }
        return;
    }

    for (LoopVar i = 0; i < inputN; i++) {
        for (LoopVar j = 0; j < cValidLen; j++) {
            for (LoopVar k = 0; k < inputH; k++) {
                uint64_t tmpDstStride = i * tmpDstStride0 + j * tmpDstStride1 + k * tmpDstStride2;
                uint64_t DstStride = (n + i) * dstStride0 + (dstCStart + j) * dstStride1 + (h + k) * dstStride2 + w;
                pto::TASSIGN(tmpDstTile, (uint64_t)(tmpDstAddr + tmpDstStride));
                GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, 1, inputW),
                                      pto::Stride(1, 1, 1, inputW, 1));
                pto::TSTORE(globalData, tmpDstTile);
            }
        }
    }
}

#define OP_TILE_OP_TRANSDATA_NCDHW2NDC1HWC0 TTransDataNCDHW2NDC1HWC0
template <int N, int D, int C, int H, int W, typename DST, typename TYPEC, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCDHW2NDC1HWC0(DST dst, TYPEC coordinate, TMP tmpTensor, INPUT input, int n, int d,
                                                int c, int h, int w)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileD = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;

    constexpr int elementSize = tileD * tileC * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<tileD, tileC, tileH, tileW>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                         pto::ConvTileShape<tileD, tileC1, tileH, tileW, C0>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaAddr = tmpDstAddr + elementSize;
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);

    auto inputAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(input.GetAddr()));
    const auto inputLayout = input.GetLayout();
    const auto gmLayout = dst.GetLayout();

    constexpr auto tmpDstStride0 = tileC1 * tileH * tileW * C0;
    constexpr auto tmpDstStride1 = tileH * tileW * C0;
    constexpr auto tmpDstStride2 = tileW * C0;
    constexpr auto tmpDstStride3 = C0;

    const auto dstStride0 = gmLayout.template GetStrideDim<DIM_1ST, 6>();
    const auto dstStride1 = gmLayout.template GetStrideDim<DIM_2ND, 6>();
    const auto dstStride2 = gmLayout.template GetStrideDim<DIM_3RD, 6>();
    const auto dstStride3 = gmLayout.template GetStrideDim<DIM_4TH, 6>();
    const auto dstStride4 = gmLayout.template GetStrideDim<DIM_5TH, 6>();

    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 6>(coordinate));
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputD = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputC1 = (inputC + C0 - 1) / C0;

    using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileW, C0, pto::BLayout::RowMajor, -1, -1>;
    using GlobalData = pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>,
                                         pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDstTile(inputW, C0);

    for (LoopVar loopN = 0; loopN < tileN; loopN++) {
        pto::TASSIGN(convInput, (uint64_t)(inputAddr + loopN * elementSize));

        Sync23_VS();
        pipe_barrier(PIPE_ALL);
        pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
        SyncVS_3();
        pipe_barrier(PIPE_ALL);

        for (LoopVar i = 0; i < inputD; i++) {
            for (LoopVar j = 0; j < inputC1; j++) {
                for (LoopVar k = 0; k < inputH; k++) {
                    uint64_t tmpDstStride = i * tmpDstStride0 + j * tmpDstStride1 + k * tmpDstStride2;
                    uint64_t DstStride = (n + loopN) * dstStride0 + (d + i) * dstStride1 + (c / C0 + j) * dstStride2 +
                                         (h + k) * dstStride3 + w * dstStride4;
                    pto::TASSIGN(tmpDstTile, (uint64_t)(tmpDstAddr + tmpDstStride));
                    GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, inputW, C0),
                                          pto::Stride(1, 1, 1, C0, 1));
                    pto::TSTORE(globalData, tmpDstTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_TRANSDATA_NDC1HWC02NCDHW TTransDataNDC1HWC02NCDHW
template <int N, int D, int dstC, int H, int W, typename DST, typename TYPEC, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNDC1HWC02NCDHW(DST dst, TYPEC coordinate, TMP tmpTensor, INPUT input, int n, int d,
                                                int c1, int h, int w, int c0, int groupIndex, int group, int padSize)
{
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto tileD = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto C0 = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;

    auto perDstGroupC = dstC / group;
    auto perDstGroupC1 = (perDstGroupC + C0 - 1) / C0;
    auto dstCStart = groupIndex * perDstGroupC + (c1 % perDstGroupC1) * C0;

    constexpr int elementSize = tileD * tileC1 * tileH * tileW * C0;
    constexpr int bufferSize = elementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<tileD, tileC1, tileH, tileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<tileD, tileC1 * C0, tileH, tileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                  tileH * tileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaAddr = tmpDstAddr + elementSize;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);

    Sync2_VS();
    pipe_barrier(PIPE_ALL);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
    SyncVS_3();
    pipe_barrier(PIPE_ALL);

    const auto inputLayout = input.GetLayout();
    const auto gmLayout = dst.GetLayout();

    constexpr auto tmpDstStride0 = tileD * tileC1 * C0 * tileH * tileW;
    constexpr auto tmpDstStride1 = tileC1 * C0 * tileH * tileW;
    constexpr auto tmpDstStride2 = tileH * tileW;
    constexpr auto tmpDstStride3 = tileW;

    auto dstStride0 = gmLayout.template GetStrideDim<DIM_1ST, 5>();
    auto dstStride1 = gmLayout.template GetStrideDim<DIM_2ND, 5>();
    auto dstStride2 = gmLayout.template GetStrideDim<DIM_3RD, 5>();
    auto dstStride3 = gmLayout.template GetStrideDim<DIM_4TH, 5>();

    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    DstAddr = DstAddr + n * dstStride0;
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 5>(coordinate));

    auto inputD = inputLayout.template GetShapeDim<DIM_1ST, 5>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, 5>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, 5>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, 5>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, 5>();
    constexpr auto realTileW = (tileW + C0 - 1) / C0 * C0;
    using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, 1, realTileW, pto::BLayout::RowMajor, -1,
                                 -1>;
    using GlobalData = pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>,
                                         pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDstTile(1, inputW);
    auto cValidLen = inputC1 * C0 - padSize;

    // 处理尾部数据
    if (tileW % C0 != 0) {
        for (LoopVar l = 0; l < inputD; l++) {
            for (LoopVar j = 0; j < cValidLen; j++) {
                for (LoopVar k = 0; k < inputH; k++) {
                    uint64_t tmpDstStride = l * tmpDstStride1 + j * tmpDstStride2 + k * tmpDstStride3;
                    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                    for (LoopVar m = 0; m < inputW; m++) {
                        tmpAreaAddr[m] = tmpDstAddr[tmpDstStride + m];
                    }
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                    pto::TASSIGN(tmpDstTile, (uint64_t)(tmpAreaAddr));
                    uint64_t DstStride = (d + l) * dstStride1 + (dstCStart + j) * dstStride2 + (h + k) * dstStride3 + w;
                    GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, 1, inputW),
                                          pto::Stride(1, 1, 1, inputW, 1));
                    pto::TSTORE(globalData, tmpDstTile);
                }
            }
        }
        return;
    }

    for (LoopVar l = 0; l < inputD; l++) {
        for (LoopVar j = 0; j < cValidLen; j++) {
            for (LoopVar k = 0; k < inputH; k++) {
                uint64_t tmpDstStride = l * tmpDstStride1 + j * tmpDstStride2 + k * tmpDstStride3;
                uint64_t DstStride = (d + l) * dstStride1 + (dstCStart + j) * dstStride2 + (h + k) * dstStride3 + w;
                pto::TASSIGN(tmpDstTile, (uint64_t)(tmpDstAddr + tmpDstStride));
                GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, 1, inputW),
                                      pto::Stride(1, 1, 1, inputW, 1));
                pto::TSTORE(globalData, tmpDstTile);
            }
        }
    }
}

#define OP_TILE_OP_TRANSDATA_NCDHW2FRACTAL_Z_3D TTransDataNCDHW2FRACTAL_Z_3D
template <int N, int C, int D, int H, int W, typename DST, typename TYPEC, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCDHW2FRACTAL_Z_3D(DST dst, TYPEC coordinate, TMP tmpTensor, INPUT input, int n, int c,
                                                    int d, int h, int w, int groupIdx, int group)
{
    n = n % N;
    c = c % C;
    d = d % D;
    h = h % H;
    w = w % W;
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = 32 / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto tileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto tileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto tileD = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto tileH = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;
    constexpr auto tileC1 = tileC / C0;
    constexpr auto tileN1 = tileN / N0;
    constexpr int elementSize = tileN * tileC * tileD * tileH * tileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCDHW,
                                        pto::ConvTileShape<tileN, tileC, tileD, tileH, tileW>>;
    using tmpDst1TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize,
                                          pto::Layout::FRACTAL_Z_3D,
                                          pto::ConvTileShape<tileD * tileC1 * tileH * tileW, tileN1, N0, C0>>;
    using tmp1TileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tileW, C0, pto::BLayout::RowMajor,
                                   tileH * tileW, C0>;
    inputTileData convInput;
    tmpDst1TileData convTmpDstNC1HWC0;
    tmp1TileData tmpTile;

    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaTileAddr = tmpDstAddr + elementSize;
    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDstNC1HWC0, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpTile, (uint64_t)tmpAreaTileAddr);

    Sync2_VS();
    pipe_barrier(PIPE_ALL);
    pto::TTRANS(convTmpDstNC1HWC0, convInput, tmpTile);
    SyncVS_3();
    pipe_barrier(PIPE_ALL);

    const auto inputLayout = input.GetLayout();
    const auto gmLayout = dst.GetLayout();
    constexpr auto tmpDstStride0 = tileN1 * N0 * C0;
    constexpr auto tmpDstStride1 = N0 * C0;
    constexpr auto tmpDstStride2 = C0;

    const auto dstStride0 = gmLayout.template GetStrideDim<DIM_1ST, 4>();
    const auto dstStride1 = gmLayout.template GetStrideDim<DIM_2ND, 4>();
    const auto dstStride2 = gmLayout.template GetStrideDim<DIM_3RD, 4>();

    auto DstAddr = (__gm__ typename DST::Type*)((uint64_t)(dst.GetAddr()));
    DstAddr = DstAddr + groupIdx * N * C * D * H * W;
    size_t gmOffset = static_cast<size_t>(gmLayout.template GetGmOffset<TYPEC, 5>(coordinate));
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputD = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputC1 = (inputC + C0 - 1) / C0;
    auto inputN1 = (inputN + N0 - 1) / N0;
    using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, N0, C0, pto::BLayout::RowMajor, -1, -1>;
    using GlobalData = pto::GlobalTensor<typename DST::Type, pto::Shape<-1, -1, -1, -1, -1>,
                                         pto::Stride<-1, -1, -1, -1, -1>>;
    TileDefine tmpDst2Tile(N0, C0);

    int64_t offsetC1 = c / C0;
    int64_t offsetN1 = n / N0;
    constexpr int64_t C1 = C / C0;

    for (LoopVar l = 0; l < inputD; l++) {
        for (LoopVar i = 0; i < inputC1; i++) {
            for (LoopVar k = 0; k < inputH; k++) {
                int64_t idx = (d + l) * C1 * H * W + (offsetC1 + i) * H * W + (h + k) * W + w;
                for (LoopVar m = 0; m < inputW; m++) {
                    int64_t tmpDst2Idx = l * tileC1 * tileH * tileW + i * tileH * tileW + k * tileW + m;
                    for (LoopVar j = 0; j < inputN1; j++) {
                        uint64_t tmpDst2Stride = tmpDst2Idx * tmpDstStride0 + j * tmpDstStride1;
                        uint64_t DstStride = idx * dstStride0 + (j + offsetN1) * dstStride1;
                        pto::TASSIGN(tmpDst2Tile, (uint64_t)(tmpDstAddr + tmpDst2Stride));
                        GlobalData globalData(DstAddr + gmOffset + DstStride, pto::Shape(1, 1, 1, N0, C0),
                                              pto::Stride(1, 1, 1, C0, 1));
                        pto::TSTORE(globalData, tmpDst2Tile);
                    }
                    idx++;
                }
            }
        }
    }
}
#endif
