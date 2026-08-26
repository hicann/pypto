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
 * \file conv_bp_pto.h
 * \brief Conv Backward TileOp Interface Definition
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_BP_DX_LOAD2D_LOAD3D_PTO__H
#define TILEOP_TILE_OPERATOR_CONV_BP_DX_LOAD2D_LOAD3D_PTO__H

#include "impl/conv_utils.h"

template <bool isConv3D, typename T, typename U>
TILEOP void TLoadConvBpDxDedy(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                              const int64_t& offset3, const int64_t& offset4, const int64_t& shape0,
                              const int64_t& shape1, const int64_t& shape2, const int64_t& shape3,
                              const int64_t& shape4, const int64_t& strideH, const int64_t& strideW,
                              const int64_t& skipH, const int64_t& skipW)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    // dstAL1 永远是 NC1HWC0 (5维): H=idx2, W=idx3
    constexpr auto stcDstH = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstW = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    constexpr int64_t rowSize = (stcDstH * stcDstW + 16 - 1) / 16 * 16;

    // 2D: GM stride [N, C1, H, W], D 轴补 stride=0; 3D: GM stride [N, D, C1, H, W]
    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = isConv3D ? GetConvStride<CONV_IDX_1>(src) : 0;
    int64_t srcStride2 = GetConvStride<CONV_IDX_1 + (isConv3D ? 1 : 0)>(src);
    int64_t srcStride3 = GetConvStride<CONV_IDX_2 + (isConv3D ? 1 : 0)>(src);
    int64_t srcStride4 = GetConvStride<CONV_IDX_3 + (isConv3D ? 1 : 0)>(src);

    using shapeDim = pto::Shape<-1, -1, -1, -1, -1>;
    using strideDim = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NDC1HWC0>;
    constexpr auto bufferSize = stcDstW * BLOCK_ALIGN_BYTE;
    using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NDC1HWC0,
                                   pto::ConvTileShape<1, 1, 1, 1, 1, c0Size>>;

    constexpr auto stcDstC1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    using zeroTileData = pto::Tile<pto::TileType::Mat, typename T::Type, rowSize, stcDstC1 * c0Size,
                                   pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    zeroTileData zeroTile(0, 0);
    pto::TASSIGN(zeroTile, (uint64_t)dst.GetAddr());
    pto::TFILLPAD(zeroTile, zeroTile);
    pipe_barrier(PIPE_MTE2);

    for (int64_t srcCout1 = 0; srcCout1 < shape2; srcCout1++) {
        for (int64_t srcH = 0; srcH < shape3; srcH++) {
            for (int64_t srcW = 0; srcW < shape4; srcW++) {
                int64_t gmOffset = offset0 * srcStride0 + offset1 * srcStride1 + (offset2 + srcCout1) * srcStride2 +
                                   (offset3 + srcH) * srcStride3 + (offset4 + srcW) * srcStride4;
                int64_t dstStride = srcCout1 * dstH * dstW + strideH * dstW * srcH + strideW * srcW;
                globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim(1, 1, 1, 1, 1),
                                     strideDim(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4));
                tileData dstL1;
                pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr() +
                                        (skipH * dstW + skipW + dstStride) * c0Size * sizeof(typename T::Type));
                pto::TLOAD(dstL1, srcGlobal);
            }
        }
    }
}

template <typename T, typename U>
INLINE void TLoadConvBPNZ(T& dst, U& src, int64_t offset0, int64_t offset1, int64_t offset2, int64_t offset3)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto stcDstBlockCol = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstBlockRow = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstFractalRow = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;

    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = GetConvStride<CONV_IDX_1>(src);

    int64_t blockCol = GetConvShape<CONV_IDX_0>(dst);
    int64_t blockRow = GetConvShape<CONV_IDX_1>(dst);
    int64_t fractalRow = GetConvShape<CONV_IDX_2>(dst);

    using shapeDim = pto::Shape<1, -1, -1, 16, c0Size>;
    using strideDim = pto::Stride<1, -1, c0Size * 16, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NZ>;
    int64_t gmOffset = offset0 * srcStride0 + offset1 * srcStride1;

    globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr()) + gmOffset, shapeDim(blockCol, blockRow),
                         strideDim(srcStride0));

    // [blockCol(dst0),blockRow(dst1),fractalRow(dst2),c0]
    // 如果是FZ格式，则为[C1HW,N1,N0,C0]
    constexpr auto stcRow = stcDstBlockRow * stcDstFractalRow;
    constexpr auto stcCol = stcDstBlockCol * c0Size;
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, stcRow, stcCol, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor>;

    tileData dstL1(blockRow * fractalRow, blockCol * c0Size);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, srcGlobal);
}

template <typename T, typename U>
TILEOP void TLoad2DDX(T& dst, U& src, const int64_t& kL0Size, const int64_t& nL0Size, const int64_t& hwk,
                      const int64_t& k0Idx, const int64_t& n0Idx)
{
    // L1: FRACTAL_Z  [Cin1*kh*kw, Cout1, Cout0, Cin0] = [c1hw, n1, n0, c0]
    //     cbuf: c1hw groups × (n1 × c0 columns of n0=16 elem each, ColMajor)
    // L0:  TileRight [K = Cout1*kh*kw*Cout0, N = Cin1*Cin0]
    constexpr auto staticN0 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    int64_t c1hw = GetConvShape<CONV_IDX_0>(src); // Cin1*kh*kw
    int64_t n1 = GetConvShape<CONV_IDX_1>(src);   // Cout1
    int64_t n0 = GetConvShape<CONV_IDX_2>(src);   // Cout0 = 16
    int64_t c0 = GetConvShape<CONV_IDX_3>(src);   // Cin0   32 / dtype
    int64_t kL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t nL0 = GetConvShape<CONV_IDX_1>(dst);

    int64_t kCout0Num = k0Idx / n0; // L1B 在K方向上小块[n0,c0]的起始Idx
    int64_t kRepeat = kL0Size / n0;
    int64_t nCin0Num = n0Idx / c0;
    int64_t nRepeat = nL0Size / c0;

    int64_t srcB1Offset;
    int64_t dstB0Offset;

    using srcL1 = pto::Tile<pto::TileType::Mat, typename U::Type, staticN0, staticC0, pto::BLayout::ColMajor, -1, -1,
                            pto::SLayout::RowMajor>;

    using dstL0 = pto::TileRight<typename T::Type, staticN0, staticC0, -1, -1>;

    for (int64_t j = 0; j < nRepeat; j++) {
        for (int64_t i = 0; i < kRepeat; i++) {
            int64_t iRev = kCout0Num + i;  // 计算在k方向上是第几个[n0,c0]小块
            int64_t hwRevIdx = iRev % hwk; // 计算在第cout1Idx个N1块上hw个[n0,co]小块中的第几个
            int64_t hwSrcIdx = hwk - hwRevIdx - 1;
            int64_t n1Idx = iRev / hwk;
            srcB1Offset = (nCin0Num + j) * n0 * c0 * n1 * hwk + hwSrcIdx * n0 * c0 * n1 + n1Idx * n0 * c0;
            dstB0Offset = i * n0 * nL0 + j * n0 * c0;
            srcL1 l1(n0, c0);
            dstL0 l0(n0, c0);
            pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()) + srcB1Offset * sizeof(typename U::Type));
            pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()) + dstB0Offset * sizeof(typename T::Type));

            pto::TEXTRACT<dstL0, srcL1>(l0, l1, 0, 0);
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_CONV_BP_DX_LOAD2D_LOAD3D_PTO__H
