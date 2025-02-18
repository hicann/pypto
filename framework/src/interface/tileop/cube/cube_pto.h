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
 * \file cube_pto.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_PTO__H
#define TILEOP_TILE_OPERATOR_CUBE_PTO__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace TileOp {
constexpr int16_t SHAPE_DIM2 = 2;

template <bool enableNZ2ND, bool isAcc, uint8_t reluMode>
struct TStoreConfig {
    static constexpr bool kEnableNZ2ND = enableNZ2ND;
    static constexpr bool kIsAcc = isAcc;
    static constexpr uint8_t kReluMode = reluMode;
};

template <int16_t idx, typename U>
INLINE int64_t GetShape(const U &tileTensor)
{
    static_assert(idx < SHAPE_DIM2, "Idx should be less than 2");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetShapeDim<idx>();
}

template <int16_t idx, typename U>
INLINE int64_t GetStride(const U &tileTensor)
{
    static_assert(idx < SHAPE_DIM2, "Idx should be less than 2");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetStrideDim<idx>();
}

INLINE int64_t CalNZOffset(const int64_t &srcShape0, const int64_t &srcShape1, const int64_t &offset0,
                           const int64_t &offset1, const int64_t &c0Size)
{
    int64_t batchSize = srcShape0 * srcShape1;
    int64_t offsetElem = offset1 + offset0 * srcShape1;
    int64_t batchIndex = offsetElem / batchSize;
    int64_t gmOffset = batchIndex * batchSize + (offset1 * srcShape0) + (offset0 - batchIndex * srcShape0) * c0Size;
    return gmOffset;
}

template <CopyInMode mode, typename Coord, typename T, typename U>
TILEOP void TLoad(T &dst, U &src, const Coord &coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    auto offset0 = coord.GetValue();
    auto offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();

    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
                  "[TLoad Error]: Dst format shoulde be L1 and Src format shoulde be GM");
    if constexpr (mode == CopyInMode::ND2NZ) {
        TLoadND2NZ(dst, src, offset0, offset1);
    } else if (mode == CopyInMode::NZ2NZ) {
        TLoadNZ2NZ(dst, src, offset0, offset1);
    } else if (mode == CopyInMode::ND2ND){
        TLoadND2ND(dst, src, offset0, offset1);
    }
    return;
}

template <typename T, typename U>
INLINE void TLoadND2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    auto gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
                          pto::Shape<1, 1, 1, -1, -1>(staticL1H, staticL1W),
                          pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
                               -1, SLayout::RowMajor>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

template <typename T, typename U>
INLINE void TLoadNZ2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    using shapeDim2 = pto::Shape<1, -1, -1, c0Size, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    int64_t gmOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
    globalData src0Global(
        (__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        pto::Shape<1, -1, -1, c0Size, c0Size>(staticL1W / c0Size, staticL1H / c0Size),
        pto::Stride<-1, -1, -1, c0Size, 1>(srcShape0 * srcShape1, srcShape0 * c0Size, c0Size * c0Size));
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1,
                               -1, SLayout::RowMajor>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

template <typename T, typename U>
INLINE void TLoadND2ND(T &dst, U &src, const int64_t &offset0, const int64_t &offset1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcStride0 = GetStride<0>(src);
    int64_t srcStride1 = GetStride<1>(src);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    auto gmOffset = offset1 + offset0 * srcShape1;
    globalData src0Global((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
                          pto::Shape<1, 1, 1, -1, -1>(staticL1H, staticL1W),
                          pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));
    using tileData =
        pto::Tile<pto::TileType::Mat, typename T::Type, staticL1H, staticL1W, pto::BLayout::RowMajor, -1, -1>;
    tileData dstL1(dstShape0, dstShape1);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, src0Global);
    return;
}

template <bool isTrans, typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    // L12L0
    if constexpr ((T::FORMAT == Hardware::L0A || T::FORMAT == Hardware::L0B) && U::FORMAT == Hardware::L1) {
        constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
        constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
        auto offset0 = coord.GetValue();
        auto offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
        constexpr auto staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
        constexpr auto staticL0W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
        using tileL1Tensor =
            pto::Tile<pto::TileType::Mat, typename U::Type, isTrans ? staticL1W : staticL1H,
                      isTrans ? staticL1H : staticL1W, isTrans ? pto::BLayout::RowMajor : pto::BLayout::ColMajor,
                      isTrans ? staticL1W : staticL1H, isTrans ? staticL1H : staticL1W,
                      isTrans ? pto::SLayout::ColMajor : pto::SLayout::RowMajor>;
        using tileL0Tensor = std::conditional_t<T::FORMAT == Hardware::L0A,
                                                pto::TileLeft<typename T::Type, staticL0H, staticL0W>,
                                                pto::TileRight<typename T::Type, staticL0H, staticL0W>>;
        tileL1Tensor l1Tile;
        tileL0Tensor l0Tile;
        int32_t srcOffset = CalNZOffset(staticL1H, staticL1W, offset0, offset1, c0Size);
        pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr() + srcOffset);
        pto::TASSIGN(l0Tile, (uint64_t)dst.GetAddr());
        pto::TEXTRACT(l0Tile, l1Tile);
        return;
    }
    if constexpr ((T::FORMAT == Hardware::BIAS || T::FORMAT == Hardware::FIXBUF) && U::FORMAT == Hardware::L1) {
        constexpr auto staticL0BW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
        using tileL1Tensor =
            pto::Tile<pto::TileType::Mat, typename T::Type, 1, staticL0BW, pto::BLayout::RowMajor, 1, staticL0BW>;
        using tileBiasOrFbTensor = pto::Tile<T::FORMAT == Hardware::BIAS ? TileType::Bias : TileType::Scaling,
                                             typename T::Type, 1, staticL0BW, BLayout::RowMajor, 1, staticL0BW>;
        tileL1Tensor l1Tensor;
        tileBiasOrFbTensor biasOrFbTensor;
        pto::TMOV(biasOrFbTensor, l1Tensor);
        return;
    }
    return;
}

template <bool enableNZ2ND, typename Coord, typename T, typename U>
TILEOP void TExtract(T &dst, U &src, const Coord &coord, int16_t subblockId)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::L0C) {
        auto offset0 = coord.GetValue();
        auto offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
        constexpr auto staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
        constexpr auto staticUBW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
        constexpr auto staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
        constexpr auto staticL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
        int64_t srcShape0 = GetShape<0>(src);
        int64_t srcShape1 = GetShape<1>(src);
        int64_t l0cOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);
        using tileUBTensor =
            pto::Tile<pto::TileType::Vec, typename T::Type, staticUBH, staticUBW, pto::BLayout::RowMajor, staticUBH,
                      staticUBW, enableNZ2ND ? pto::SLayout::NoneBox : pto::SLayout::ColMajor>;
        using tileL0CTensor = pto::TileAcc<typename U::Type, staticL0CH, staticL0CW>;
        tileUBTensor UBTile;
        tileL0CTensor l0cTile;
        pto::TASSIGN(UBTile, (uint64_t)dst.GetAddr() + l0cOffset);
        pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr());
        if (subblockId == 0) {
            pto::TMOV<tileUBTensor, tileL0CTensor, AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
        } else {
            pto::TMOV<tileUBTensor, tileL0CTensor, AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
        }
    }
}

template <bool isZeroC, typename T, typename U, typename V>
TILEOP void Matmul(T &c, U &a, V &b)
{
    constexpr auto shapeSizeA = Std::tuple_size<typename U::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename V::Shape>::value;
    constexpr auto shapeSizeC = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[Matmul ERROR]: Shape dim size shoulde be 2");

    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename U::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename V::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename V::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T::TileShape>::type::value;

    using tileL0ATensor = pto::TileLeft<typename U::Type, staticL0AH, staticL0AW>;
    using tileL0BTensor = pto::TileRight<typename V::Type, staticL0BH, staticL0BW>;
    using tileL0CTensor = pto::TileAcc<typename T::Type, staticL0CH, staticL0CW>;

    tileL0ATensor l0a;
    tileL0BTensor l0b;
    tileL0CTensor l0c;

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());

    if constexpr (!isZeroC) {
        pto::TMATMUL(l0c, l0a, l0b);
    } else {
        pto::TMATMUL_ACC(l0c, l0c, l0a, l0b);
    }
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void Matmul(T0 &c, T1 &a, T2 &b, T3 &bias)
{
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[Matmul ERROR]: Shape dim size shoulde be 2");

    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T2::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T2::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;

    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW>;
    using tileL0BTensor = pto::TileRight<typename T2::Type, staticL0BH, staticL0BW>;
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW>;
    using tileBiasTensor =
        pto::Tile<TileType::Bias, typename T3::Type, 1, staticL0BW, BLayout::RowMajor, 1, staticL0BW>;

    tileL0ATensor l0a;
    tileL0BTensor l0b;
    tileL0CTensor l0c;
    tileBiasTensor biasT;

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());
    pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
    pto::TMATMUL_BIAS(l0c, l0a, l0b, biasT);
}

template <typename config, typename T, typename U>
INLINE void TStoreNZ2ND(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);

    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    int64_t gmOffset = offset1 + offset0 * dstShape1;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
                         pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1),
                         pto::Stride<1, 1, 1, -1, -1>(dstStride0, dstStride1));
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1,
                               SLayout::RowMajor>;
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TSTORE(dstGlobal, srcL0C);
    return;
}

template <typename config, typename T, typename U>
INLINE void TStoreNZ2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);

    constexpr auto tileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    int64_t gmOffset = CalNZOffset(dstShape0, dstShape1, offset0, offset1, c0Size);
    using shapeDim2 = pto::Shape<1, -1, -1, c0Size, c0Size>;
    using strideDim2 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim2, strideDim2, pto::Layout::NZ>;
    globalData dstGlobal(
        (__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        pto::Shape<1, -1, -1, c0Size, c0Size>(dstShape1 / c0Size, dstShape0 / c0Size),
        pto::Stride<-1, -1, -1, c0Size, 1>(dstShape0 * dstShape1, dstShape0 * c0Size, c0Size * c0Size));
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, tileH, tileW, pto::BLayout::ColMajor, -1, -1,
                               SLayout::RowMajor>;
    tileData srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TSTORE(dstGlobal, srcL0C);
    return;
}

template <typename config, typename Coord, typename T, typename U>
TILEOP void TStore(T &dst, U &src, const Coord &coord, uint64_t scaleValue = 0)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM2 && Std::tuple_size<Coord>::value == SHAPE_DIM2, "Shape Size should be 2 Dim");
    auto offset0 = coord.GetValue();
    auto offset1 = static_cast<const Std::tuple<size_t> &>(coord).GetValue();
    if constexpr (U::FORMAT == Hardware::L0C && T::FORMAT == Hardware::GM) {
        if constexpr (config::kEnableNZ2ND) {
            TStoreNZ2ND<config>(dst, src, offset0, offset1, scaleValue);
            return;
        } else {
            TStoreNZ2NZ<config>(dst, src, offset0, offset1, scaleValue);
            return;
        }
    }
}
}  // namespace TileOp
#endif  // TILEOP_TILE_OPERATOR_CUBE_PTO__H