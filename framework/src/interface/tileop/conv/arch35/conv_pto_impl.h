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
 * \file conv_pto_impl.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_PTO_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_PTO_IMPL__H
#include <limits.h>
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "../common/conv_load2d_load3d_impl.h"


/**
 * Calculate load GM offset for input/weight with NCHW format.
 * shapeInfo: input -> [orgCi, orgHi, orgWi], weight -> [orgCi, kh, kw]
 * shapeInfo: input -> [  0  ,   1  ,   2  ], weight -> [  0  , 1 , 2 ]
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_h_offset, weight -> 0 (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: input -> src_w_offset, weight -> 0
 * offset4: input -> 0 (占位), weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap>
INLINE int64_t CalLoadOffsetNCHW(const ShapeInfo& shapeInfo, const OffsetInfo& offsetInfo)
{
    if constexpr (isFmap) {
        int64_t inputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2;
        int64_t offsetC = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2;
        int64_t offsetH = offsetInfo.offset2 < 0 ? 0 : offsetInfo.offset2;
        offsetH = offsetInfo.offset2 > shapeInfo.shape1 ? shapeInfo.shape1 : offsetInfo.offset2;
        int64_t offsetW = offsetInfo.offset3 < 0 ? 0 : offsetInfo.offset3;
        offsetW = offsetInfo.offset3 > shapeInfo.shape2 ? shapeInfo.shape2 : offsetInfo.offset3;
        return offsetInfo.offset0 * inputOneBatchSize + offsetC + offsetH * shapeInfo.shape2 + offsetW;
    } else {
        return offsetInfo.offset0 * shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2 +
               offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2;
    }
}

/**
 * Calculate load GM offset for input/weight with NCDHW format.
 * shapeInfo: input -> [orgCi, orgDi, orgHi, orgWi], weight -> [orgCi, kd, kh, kw]
 * shapeInfo: input -> [  0  ,   1  ,   2  ,   3  ], weight -> [  0  , 1 , 2 , 3 ]
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_d_offset, weight -> src_d_offset
 * offset3: input -> src_h_offset, weight -> 0
 * offset4: input -> src_w_offset, weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap>
INLINE int64_t CalLoadOffsetNCDHW(const ShapeInfo& shapeInfo, const OffsetInfo& offsetInfo)
{
    if constexpr (isFmap) {
        int64_t inputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
        int64_t offsetC = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
        int64_t offsetD = offsetInfo.offset2 * shapeInfo.shape2 * shapeInfo.shape3;
        int64_t offsetH = offsetInfo.offset3 < 0 ? 0 : offsetInfo.offset3;
        offsetH = offsetInfo.offset3 > shapeInfo.shape2 ? shapeInfo.shape2 : offsetInfo.offset3;
        int64_t offsetW = offsetInfo.offset4 < 0 ? 0 : offsetInfo.offset4;
        offsetW = offsetInfo.offset4 > shapeInfo.shape3 ? shapeInfo.shape3 : offsetInfo.offset4;
        return offsetInfo.offset0 * inputOneBatchSize + offsetC + offsetD + offsetH * shapeInfo.shape3 + offsetW;
    } else {
        int64_t khxkw = shapeInfo.shape2 * shapeInfo.shape3;
        int64_t kdxkhxkw = shapeInfo.shape1 * khxkw;
        return offsetInfo.offset0 * shapeInfo.shape0 * kdxkhxkw + offsetInfo.offset1 * kdxkhxkw +
               offsetInfo.offset2 * khxkw;
    }
}

/**
 * Calculate store GM offset with NZ -> NCHW format.
 * shapeInfo: [cout, dout, hout, wout]
 * shapeInfo: [  0 ,  1  ,  2  ,  3  ]
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
INLINE int64_t CalStoreOffsetNCDHW(const ShapeInfo& shapeInfo, const OffsetInfo& offsetInfo, const int64_t& loopH)
{
    int64_t outputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
    int64_t coutOffset = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2 * shapeInfo.shape3;
    int64_t doutOffset = offsetInfo.offset2 * shapeInfo.shape2 * shapeInfo.shape3;
    return offsetInfo.offset0 * outputOneBatchSize + coutOffset + doutOffset +
           (offsetInfo.offset3 + loopH) * shapeInfo.shape3 + offsetInfo.offset4;
}

/**
 * Copy input data from DDR to L1 with DN2NZ, input NCHW -> NC1HWC0, weight NCHW -> FZ.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NCHW), weigh -> GM(NCHW)
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_h_offset, weight -> 0 (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: input -> src_w_offset, weight -> 0
 * offset4: input -> 0 (占位), weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap, typename T, typename U>
INLINE void TLoadConv2DDN2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcC = GetConvShape<CONV_IDX_1>(src);
    int64_t srcH = GetConvShape<CONV_IDX_2>(src);
    int64_t srcW = GetConvShape<CONV_IDX_3>(src);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    int64_t srcStrideN = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStrideC = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStrideH = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStrideW = GetConvStride<CONV_IDX_3>(src);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    ShapeInfo shapeInfo = {srcC, srcH, srcW};
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    int64_t gmOffset = CalLoadOffsetNCHW<isFmap>(shapeInfo, offsetInfo);
    // conv2d 参数顺序调整：n, c, h, w, 0 (最后的0占位)
    // srcShapeInfo: {shape0(n), shape1(c), shape2(h), shape3(w), shape4(0)}
    globalData srcGlobal(
        (__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
        shapeDim4(srcShapeInfo.shape0, srcShapeInfo.shape1, srcShapeInfo.shape2, srcShapeInfo.shape3),
        strideDim4(srcStrideN, srcStrideC, srcStrideH, srcStrideW));
    if constexpr (isFmap) {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<
            pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NC1HWC0,
            pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<
            pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::FRACTAL_Z,
            pto::ConvTileShape<-1, -1, -1, -1, 1>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
}

/**
 * Copy input data from DDR to L1 with DN2NZ, input NCDHW -> NDC1HWC0, weight NCDHW -> FZ_3D.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NCHW), weigh -> GM(NCHW)
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_d_offset, weight -> src_d_offset
 * offset3: input -> src_h_offset, weight -> 0
 * offset4: input -> src_w_offset, weight -> 0
 * isFmap: true -> input, false -> weight
 */
template <bool isFmap, typename T, typename U>
INLINE void TLoadConv3DDN2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcC = GetConvShape<CONV_IDX_1>(src);
    int64_t srcD = GetConvShape<CONV_IDX_2>(src);
    int64_t srcH = GetConvShape<CONV_IDX_3>(src);
    int64_t srcW = GetConvShape<CONV_IDX_4>(src);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    int64_t srcStrideN = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStrideC = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStrideD = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStrideH = GetConvStride<CONV_IDX_3>(src);
    int64_t srcStrideW = GetConvStride<CONV_IDX_4>(src);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    ShapeInfo shapeInfo = {srcC, srcD, srcH, srcW};
    using shapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using strideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim5, strideDim5, pto::Layout::NCDHW>;
    int64_t gmOffset = CalLoadOffsetNCDHW<isFmap>(shapeInfo, offsetInfo);
    globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset),
        shapeDim5(srcShapeInfo.shape0, srcShapeInfo.shape1, srcShapeInfo.shape2, srcShapeInfo.shape3,
            srcShapeInfo.shape4),
        strideDim5(srcStrideN, srcStrideC, srcStrideD, srcStrideH, srcStrideW));
    if constexpr (isFmap) {
        int64_t dstShape4 = GetConvShape<CONV_IDX_4>(dst);
        constexpr auto stcDstShape4 = Std::tuple_element<CONV_IDX_4, typename T::TileShape>::type::value;
        constexpr auto bufferSize =
            stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * stcDstShape4 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<
            pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NDC1HWC0,
            pto::ConvTileShape<-1, -1, -1, -1, -1, c0Size>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3, dstShape4);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<
            pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::FRACTAL_Z_3D,
            pto::ConvTileShape<-1, -1, -1, -1, 1>>;
        tileData dstL1(dstShape0, dstShape1, dstShape2, dstShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
}

template <bool isConv3D, bool isFmap, typename T, typename U>
INLINE void TLoadConvDN2NZ(T& dst, U& src, const OffsetInfo& offsetInfo, const ShapeInfo& srcShapeInfo)
{
    if constexpr (isConv3D) {
        TLoadConv3DDN2NZ<isFmap>(dst, src, offsetInfo, srcShapeInfo);
    } else {
        TLoadConv2DDN2NZ<isFmap>(dst, src, offsetInfo, srcShapeInfo);
    }
}

/**
 * Copy data from L0C to DDR with NZ -> NCHW format.
 * dst: GM(NCHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_h_offset (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: dst_w_offset
 * offset4: 0 (占位)
 */
template <typename T, typename U>
INLINE void TStoreConv2DNZ2DN(
    T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_3>(dst);
    int64_t gmOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideC + offsetInfo.offset2 * dstW +
                       offsetInfo.offset3;

    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using tileData = pto::Tile<
        pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor,
        pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Null>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; loopH < (realM / cutW); loopH++) {
        globalData dstGlobal(
            (__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), shapeDim4(dstN, dstC, dstH, dstW),
            strideDim4(dstStrideN, dstStrideC, dstStrideH, dstStrideW));
        tileData srcL0C(cutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE(dstGlobal, srcL0C);
        gmOffset += dstW;
    }
}

/**
 * Copy data from L0C to DDR with NZ -> NCDHW format.
 * dst: GM(NCDHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
template <typename T, typename U>
INLINE void TStoreConv3DNZ2DN(
    T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstD = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_4>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideD = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_3>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_4>(dst);
    ShapeInfo shapeInfo{dstC, dstD, dstH, dstW};
    using shapeDim5 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tileData = pto::Tile<
        pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor,
        pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Null>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim5, strideDim5, pto::Layout::NCDHW>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; loopH < (realM / cutW); loopH++) {
        int64_t gmOffset = CalStoreOffsetNCDHW(shapeInfo, offsetInfo, loopH);
        globalData dstGlobal(
            (__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), shapeDim5(dstN, dstC, dstD, dstH, dstW),
            strideDim5(dstStrideN, dstStrideC, dstStrideD, dstStrideH, dstStrideW));
        tileData srcL0C(cutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE(dstGlobal, srcL0C);
    }
}

template <bool isConv3D, typename T, typename U>
INLINE void TStoreConvNZ2DN(
    T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN, const int64_t& cutW)
{
    if constexpr (isConv3D) {
        TStoreConv3DNZ2DN(dst, src, offsetInfo, realM, realN, cutW);
    } else {
        TStoreConv2DNZ2DN(dst, src, offsetInfo, realM, realN, cutW);
    }
}
#endif // TILEOP_TILE_OPERATOR_CONV_PTO_IMPL__H
