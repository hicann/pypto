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
 * \file mte.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_MTE__H
#define TILEOP_TILE_OPERATOR_MTE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T, typename U, typename C>
__aicore__ inline void TLoad(T dst, U src, C coordinate) {
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    if constexpr (T::FORMAT == Hardware::UB && U::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcShape0 = srcLayout.template GetShapeDim<0, 5>();
        auto srcShape1 = srcLayout.template GetShapeDim<1, 5>();
        auto srcShape2 = srcLayout.template GetShapeDim<2, 5>();
        auto srcShape3 = srcLayout.template GetShapeDim<3, 5>();
        auto srcShape4 = srcLayout.template GetShapeDim<4, 5>();
        auto srcStride0 = srcLayout.template GetStrideDim<0, 5>();
        auto srcStride1 = srcLayout.template GetStrideDim<1, 5>();
        auto srcStride2 = srcLayout.template GetStrideDim<2, 5>();
        auto srcStride3 = srcLayout.template GetStrideDim<3, 5>();
        auto srcStride4 = srcLayout.template GetStrideDim<4, 5>();

        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<0, 5>();
        auto dstShape1 = dstLayout.template GetShapeDim<1, 5>();
        auto dstShape2 = dstLayout.template GetShapeDim<2, 5>();
        auto dstShape3 = dstLayout.template GetShapeDim<3, 5>();
        auto dstShape4 = dstLayout.template GetShapeDim<4, 5>();
        auto dstStride0 = dstLayout.template GetStrideDim<0, 5>();
        auto dstStride1 = dstLayout.template GetStrideDim<1, 5>();
        auto dstStride2 = dstLayout.template GetStrideDim<2, 5>();
        auto dstStride3 = dstLayout.template GetStrideDim<3, 5>();
        auto dstStride4 = dstLayout.template GetStrideDim<4, 5>();
        auto gmOffset = srcLayout.template GetGmOffset<C, 5>(coordinate);
        using SrcDtype = std::conditional_t<std::is_same_v<typename U::Type, bool>, uint8_t, typename U::Type>;
        using DstDtype = std::conditional_t<std::is_same_v<typename T::Type, bool>, uint8_t, typename T::Type>;

        if constexpr (TileOp::IsConstContinous<T>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            constexpr auto tileH = TileOp::GetOutterAxisMergeResult<shapeSize, typename T::TileShape>();
            constexpr auto tileW = TileOp::GetTensorTileShapeDim<T, shapeSize - 1>();
            using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
            using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
            using GlobalData = pto::GlobalTensor<SrcDtype, ShapeDim5, StrideDim5>;
            constexpr auto constDstShape3 = TileOp::GetTensorShapeDim<T, 3, 5>();
            constexpr auto constDstShape4 = TileOp::GetTensorShapeDim<T, 4, 5>();
            GlobalData src0Global((__gm__ SrcDtype *)(src.GetAddr() + gmOffset),
                pto::Shape(dstShape0, dstShape1, dstShape2, constDstShape3, constDstShape4),
                pto::Stride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4));
            using TileData = pto::Tile<pto::TileType::Vec, DstDtype, tileH, tileW, pto::BLayout::RowMajor,
                constDstShape3, constDstShape4>;
            TileData dstUB;
            pto::TASSIGN(dstUB, (uint64_t)dst.GetAddr());
            pto::TLOAD(dstUB, src0Global);
            return;
        }

        constexpr auto tileH = TileOp::GetTensorTileShapeDim<T, 3, 5>();
        constexpr auto tileW = TileOp::GetTensorTileShapeDim<T, 4, 5>();
        for (size_t index0 = 0; index0 < dstShape0; ++index0) {
            for (size_t index1 = 0; index1 < dstShape1; ++index1) {
                for (size_t index2 = 0; index2 < dstShape2; ++index2) {
                    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
                    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
                    using GlobalData = pto::GlobalTensor<SrcDtype, ShapeDim5, StrideDim5>;
                    GlobalData src0Global((__gm__ SrcDtype *)(src.GetAddr() + gmOffset + index0 * srcStride0 +
                                                              index1 * srcStride1 + index2 * srcStride2),
                        pto::Shape(1, 1, 1, dstShape3, dstShape4), pto::Stride(0, 0, 0, srcStride3, srcStride4));
                    using TileDefine =
                        pto::Tile<pto::TileType::Vec, DstDtype, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
                    TileDefine dstUB(dstShape3, dstShape4);
                    auto ubOffset = index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;
                    pto::TASSIGN(dstUB, (uint64_t)(dst.GetAddr() + ubOffset * sizeof(DstDtype)));
                    pto::TLOAD(dstUB, src0Global);
                }
            }
        }
    }
}
template <typename T, typename U, typename C>
__aicore__ inline void TStore(T dst, U src, C coordinate) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    if constexpr (U::FORMAT == Hardware::UB && T::FORMAT == Hardware::GM) {
        const auto srcLayout = src.GetLayout();
        auto srcShape0 = srcLayout.template GetShapeDim<0, 5>();
        auto srcShape1 = srcLayout.template GetShapeDim<1, 5>();
        auto srcShape2 = srcLayout.template GetShapeDim<2, 5>();
        auto srcShape3 = srcLayout.template GetShapeDim<3, 5>();
        auto srcShape4 = srcLayout.template GetShapeDim<4, 5>();
        auto srcStride0 = srcLayout.template GetStrideDim<0, 5>();
        auto srcStride1 = srcLayout.template GetStrideDim<1, 5>();
        auto srcStride2 = srcLayout.template GetStrideDim<2, 5>();
        auto srcStride3 = srcLayout.template GetStrideDim<3, 5>();
        auto srcStride4 = srcLayout.template GetStrideDim<4, 5>();
        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<0, 5>();
        auto dstShape1 = dstLayout.template GetShapeDim<1, 5>();
        auto dstShape2 = dstLayout.template GetShapeDim<2, 5>();
        auto dstShape3 = dstLayout.template GetShapeDim<3, 5>();
        auto dstShape4 = dstLayout.template GetShapeDim<4, 5>();
        auto dstStride0 = dstLayout.template GetStrideDim<0, 5>();
        auto dstStride1 = dstLayout.template GetStrideDim<1, 5>();
        auto dstStride2 = dstLayout.template GetStrideDim<2, 5>();
        auto dstStride3 = dstLayout.template GetStrideDim<3, 5>();
        auto dstStride4 = dstLayout.template GetStrideDim<4, 5>();
        auto gmOffset = dstLayout.template GetGmOffset<C, 5>(coordinate);
        using SrcDtype = std::conditional_t<std::is_same_v<typename U::Type, bool>, uint8_t, typename U::Type>;
        using DstDtype = std::conditional_t<std::is_same_v<typename T::Type, bool>, uint8_t, typename T::Type>;

        if constexpr (TileOp::IsConstContinous<U>() == true) {
            // 对于静态整块场景，将UB合成二维，GM保持五维
            constexpr auto tileH = TileOp::GetOutterAxisMergeResult<shapeSize, typename U::TileShape>();
            constexpr auto tileW = TileOp::GetTensorTileShapeDim<U, shapeSize - 1>();
            using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
            using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
            using GlobalData = pto::GlobalTensor<DstDtype, ShapeDim5, StrideDim5>;
            constexpr auto constSrcShape3 = TileOp::GetTensorShapeDim<U, 3, 5>();
            constexpr auto constSrcShape4 = TileOp::GetTensorShapeDim<U, 4, 5>();
            GlobalData dstGlobal((__gm__ DstDtype *)(dst.GetAddr() + gmOffset),
                pto::Shape(srcShape0, srcShape1, srcShape2, constSrcShape3, constSrcShape4),
                pto::Stride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4));
            using TileData = pto::Tile<pto::TileType::Vec, SrcDtype, tileH, tileW, pto::BLayout::RowMajor,
                constSrcShape3, constSrcShape4>;
            TileData srcUB;
            pto::TASSIGN(srcUB, (uint64_t)src.GetAddr());
            pto::TSTORE(dstGlobal, srcUB);
            return;
        }

        constexpr auto tileH = TileOp::GetTensorTileShapeDim<U, 3, 5>();
        constexpr auto tileW = TileOp::GetTensorTileShapeDim<U, 4, 5>();
        for (size_t index0 = 0; index0 < srcShape0; ++index0) {
            for (size_t index1 = 0; index1 < srcShape1; ++index1) {
                for (size_t index2 = 0; index2 < srcShape2; ++index2) {
                    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
                    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
                    using GlobalData = pto::GlobalTensor<DstDtype, ShapeDim5, StrideDim5>;
                    GlobalData dstGlobal((__gm__ DstDtype *)(dst.GetAddr() + gmOffset + index0 * dstStride0 +
                                                             index1 * dstStride1 + index2 * dstStride2),
                        pto::Shape(1, 1, 1, srcShape3, srcShape4), pto::Stride(0, 0, 0, dstStride3, dstStride4));
                    using TileDefine =
                        pto::Tile<pto::TileType::Vec, SrcDtype, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
                    TileDefine srcUB(srcShape3, srcShape4);
                    auto ubOffset = index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                    pto::TASSIGN(srcUB, (uint64_t)(src.GetAddr() + ubOffset * sizeof(SrcDtype)));
                    pto::TSTORE(dstGlobal, srcUB);
                }
            }
        }
    }
}

#endif