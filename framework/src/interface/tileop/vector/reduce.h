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
 * \file reduce.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_REDUCE__H
#define TILEOP_TILE_OPERATOR_REDUCE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryOp op, typename T0, typename T1, typename T2>
TILEOP void ReduceComputeImpl(T0 dst, T1 src, T2 tmp) {
    if constexpr (op == BinaryOp::SUM) {
        pto::TROWSUM(dst, src, tmp);
        return;
    }
    if constexpr (op == BinaryOp::AMAX) {
        pto::TROWMAX(dst, src, tmp);
    }
}

template <BinaryOp op, typename T0, typename T1, typename T2>
TILEOP void ReduceCompute(T0 dst, T1 src, T2 tmp) {
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    using TmpTileDefine =
            pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstTileDefine =
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                using SrcTileDefine =
                    pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                if (srcShape3 == 0 || srcShape4 == 0){
                    return;
                }
                ReduceComputeImpl<op>(dstTile, srcTile, tmpTile);
            }
        }
    }
}

template <typename T0, typename T1, typename T2>
TILEOP void TRowSumSingle(T0 dst, T1 src, T2 tmp) {
    ReduceCompute<BinaryOp::SUM>(dst, src, tmp);
}

template <typename T0, typename T1, typename T2>
TILEOP void TRowMaxSingle(T0 dst, T1 src, T2 tmp) {
    ReduceCompute<BinaryOp::AMAX>(dst, src, tmp);
}

template <int axis, size_t srcShapeSize, size_t dstShapeSize, size_t tmpShapeSize, typename T0, typename T1, typename T2>
TILEOP void TRowSumLineStatic(T0 dst, T1 src, T2 tmp) {
    constexpr auto typeSize = sizeof(typename T1::Type);
    constexpr auto shape = TileOp::GetAnyAxisMergeResult<1, axis, typename T1::TileShape>();
    constexpr auto srcStride = TileOp::GetAnyAxisMergeResult<axis + 1, srcShapeSize, typename T1::TileShape>();
    constexpr auto dstStride = TileOp::GetAnyAxisMergeResult<axis + 1, dstShapeSize, typename T0::TileShape>();
    constexpr auto tmpStride = TileOp::GetAnyAxisMergeResult<axis + 1, tmpShapeSize, typename T2::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, axis>();
    constexpr auto srcTileW = TileOp::GetAnyAxisMergeResult<axis + 2, srcShapeSize, typename T1::TileShape>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis>();
    constexpr auto dstTileW = TileOp::GetAnyAxisMergeResult<axis + 2, dstShapeSize, typename T0::TileShape>();
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, axis>();
    constexpr auto tmpTileW = TileOp::GetAnyAxisMergeResult<axis + 2, tmpShapeSize, typename T0::TileShape>();
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type,
        srcTileH, srcTileW, pto::BLayout::RowMajor, srcTileH, srcTileW>;
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type,
        dstTileH, dstTileW, pto::BLayout::RowMajor, dstTileH, dstTileW>;
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type,
        tmpTileH, tmpTileW, pto::BLayout::RowMajor, tmpTileH, tmpTileW>;
    SrcTileDefine srcTile;
    DstTileDefine dstTile;
    TmpTileDefine tmpTile;
    for (size_t nIndex = 0; nIndex < shape; ++nIndex) {
        constexpr auto srcOffset = srcStride * nIndex;
        constexpr auto dstOffset = dstStride * nIndex;
        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
        pto::TCOLSUM(dstTile, srcTile, tmpTile, true);
    }
}

template <int axis, typename DstTileDefine, typename SrcTileDefine, typename TmpTileDefine, typename T0, typename T1, typename T2>
TILEOP void TRowSumLineDynamic(T0 dst, T1 src, T2 tmp) {
    constexpr size_t expectSize = 5;
    constexpr auto typeSize = sizeof(typename T1::Type);
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto tmpLayout = tmp.GetLayout();
    size_t dstShape[] = {
        static_cast<size_t>(dstLayout.template GetShapeDim<0, expectSize>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<1, expectSize>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<2, expectSize>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<3, expectSize>()),
        static_cast<size_t>(dstLayout.template GetShapeDim<4, expectSize>())
    };
    size_t dstStride[] = {
        static_cast<size_t>(dstLayout.template GetStrideDim<0, expectSize>()),
        static_cast<size_t>(dstLayout.template GetStrideDim<1, expectSize>()),
        static_cast<size_t>(dstLayout.template GetStrideDim<2, expectSize>()),
        static_cast<size_t>(dstLayout.template GetStrideDim<3, expectSize>())
    };
    size_t srcShape[] = {
        static_cast<size_t>(srcLayout.template GetShapeDim<0, expectSize>()),
        static_cast<size_t>(srcLayout.template GetShapeDim<1, expectSize>()),
        static_cast<size_t>(srcLayout.template GetShapeDim<2, expectSize>()),
        static_cast<size_t>(srcLayout.template GetShapeDim<3, expectSize>()),
        static_cast<size_t>(srcLayout.template GetShapeDim<4, expectSize>())
    };
    size_t srcStride[] = {
        static_cast<size_t>(srcLayout.template GetStrideDim<0, expectSize>()),
        static_cast<size_t>(srcLayout.template GetStrideDim<1, expectSize>()),
        static_cast<size_t>(srcLayout.template GetStrideDim<2, expectSize>()),
        static_cast<size_t>(srcLayout.template GetStrideDim<3, expectSize>())
    };
    size_t tmpShape[] = {
        static_cast<size_t>(tmpLayout.template GetShapeDim<0, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetShapeDim<1, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetShapeDim<2, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetShapeDim<3, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetShapeDim<4, expectSize>())
    };
    size_t tmpStride[] = {
        static_cast<size_t>(tmpLayout.template GetStrideDim<0, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetStrideDim<1, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetStrideDim<2, expectSize>()),
        static_cast<size_t>(tmpLayout.template GetStrideDim<3, expectSize>())
    };
    for (size_t n0Index = 0, n0Size = (axis == 0 ? (size_t) 1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (size_t n1Index = 0, n1Size = (axis == 1 ? (size_t) 1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (size_t n2Index = 0, n2Size = (axis == 2 ? (size_t) 1 : dstShape[2]); n2Index < n2Size; ++n2Index) {
                for (size_t n3Index = 0, n3Size = (axis == 3 ? (size_t) 1 : dstShape[3]); n3Index < n3Size; ++n3Index) {
                    DstTileDefine dstTile(dstShape[axis], dstShape[4]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[4]);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride[0] + n1Index * dstStride[1] +
                        n2Index * dstStride[2] + n3Index * dstStride[3];
                    auto srcOffset = n0Index * srcStride[0] + n1Index * srcStride[1] +
                        n2Index * srcStride[2] + n3Index * srcStride[3];
                    auto tmpOffset = n0Index * tmpStride[0] + n1Index * tmpStride[1] +
                        n2Index * tmpStride[2] + n3Index * tmpStride[3];
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    pto::TCOLSUM(dstTile, srcTile, tmpTile, true);
                }
            }
        }
    }
}

template <int axis, typename T0, typename T1, typename T2>
TILEOP void TRowSumLine(T0 dst, T1 src, T2 tmp) {
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - 5>();
    constexpr auto dstTileW = TileOp::GetAnyAxisMergeResult<axis + dstShapeSize - 3, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, axis + srcShapeSize - 5>();
    constexpr auto srcTileW = TileOp::GetAnyAxisMergeResult<axis + srcShapeSize - 3, srcShapeSize, typename T1::TileShape>();
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, tmpShapeSize - 2>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, tmpShapeSize - 1>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor, tmpTileH, tmpTileW>;
    TRowSumLineDynamic<axis, DstTileDefine, SrcTileDefine, TmpTileDefine>(dst, src, tmp);
}
#endif