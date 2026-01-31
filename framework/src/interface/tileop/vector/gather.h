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
 * \file gather.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_GATHER__H
#define TILEOP_TILE_OPERATOR_GATHER__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_GATHER_ELEMENT TgatherElement
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TgatherElement(T0 dst, T1 src0, T2 src1, T3 tmp) {
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto srcLayout = src0.GetLayout();
    auto n0SrcStride = srcLayout.template GetStrideDim<0, expectSize>();
    auto n1SrcStride = srcLayout.template GetStrideDim<1, expectSize>();
    auto n2SrcStride = srcLayout.template GetStrideDim<2, expectSize>();
    auto n3SrcStride = srcLayout.template GetStrideDim<3, expectSize>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxShape = idxLayout.template GetShapeDim<0, expectSize>();
    auto n1IdxShape = idxLayout.template GetShapeDim<1, expectSize>();
    auto n2IdxShape = idxLayout.template GetShapeDim<2, expectSize>();
    auto n3IdxShape = idxLayout.template GetShapeDim<3, expectSize>();
    auto n4IdxShape = idxLayout.template GetShapeDim<4, expectSize>();
    auto n0IdxStride = idxLayout.template GetStrideDim<0, expectSize>();
    auto n1IdxStride = idxLayout.template GetStrideDim<1, expectSize>();
    auto n2IdxStride = idxLayout.template GetStrideDim<2, expectSize>();
    auto n3IdxStride = idxLayout.template GetStrideDim<3, expectSize>();

    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<0, expectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<1, expectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<2, expectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

    constexpr bool scalarFlag = (sizeof(typename T2::Type) == 8) ? true : false;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTileShape1 = TileOp::GetOutterAxisMergeResult<shapeSize, typename T1::TileShape>();
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileShape1, srcTileW, pto::BLayout::RowMajor>;
    using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    srcTileDefine srcTile;
    idxTileDefine idxTile(1, n4IdxShape);
    dstTileDefine dstTile(1, n4IdxShape);
    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    }
    auto srcAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src0.GetAddr()));
    auto idxAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T2::Type*)((uint64_t)(tmp.GetAddr()));
    auto newIdxValue = 0;
    for (LoopVar i = 0; i < n0IdxShape; ++i) {
        for (LoopVar j = 0; j < n1IdxShape; ++j) {
            for (LoopVar k = 0; k < n2IdxShape; ++k) {
                for (LoopVar l = 0; l < n3IdxShape; ++l) {
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    }
                    for (LoopVar m = 0; m < n4IdxShape; ++m) {
                        auto dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        auto orgIdxValue = 
                            *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride + l * n3IdxStride + m);
                        if constexpr (axis == 0) {
                            newIdxValue =
                                orgIdxValue * n0SrcStride  + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride + m;
                        } else if constexpr (axis == 1) {
                            newIdxValue =
                                i * n0SrcStride  + orgIdxValue * n1SrcStride + k * n2SrcStride + l * n3SrcStride + m;
                        } else if constexpr (axis == 2) {
                            newIdxValue =
                                i * n0SrcStride  + j * n1SrcStride + orgIdxValue * n2SrcStride + l * n3SrcStride + m;
                        } else if constexpr (axis == 3) {
                            newIdxValue =
                                i * n0SrcStride  + j * n1SrcStride + k * n2SrcStride + orgIdxValue * n3SrcStride + m;
                        } else {
                            newIdxValue =
                                i * n0SrcStride  + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride + orgIdxValue;
                        }
                        if constexpr (scalarFlag) {
                            dstAddr[dstOffset] = srcAddr[newIdxValue];
                        } else {
                            *(tmpAddr + m) = newIdxValue;
                        }
                    }
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        auto dstAddrOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstAddrOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src0.GetAddr()));
                        pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                        pto::TGATHER(dstTile, srcTile, idxTile);
                    }
                }
            }
        }
    }
    if constexpr (scalarFlag) {
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}

template <int axis, size_t index0, size_t index1, size_t index2, size_t index3, size_t index4, typename T0, typename T1,
    typename T2, typename C1, typename C2>
TILEOP void Tgather(T0 dst, T1 src, T2 idx, C1 srcCoordinate, C2 idxCoordinate) {
    constexpr size_t N = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t srcExpectSize = 4;
    constexpr size_t idxExpectSize = 2;
    constexpr size_t dstExpectSize = 5;
    const auto srcLayout = src.GetLayout();
    auto n0SrcStride = srcLayout.template GetStrideDim<0, srcExpectSize>();
    auto n1SrcStride = srcLayout.template GetStrideDim<1, srcExpectSize>();
    auto n2SrcStride = srcLayout.template GetStrideDim<2, srcExpectSize>();
    auto n3SrcStride = srcLayout.template GetStrideDim<3, srcExpectSize>();

    auto n0SrcShape = srcLayout.template GetShapeDim<0, srcExpectSize>();
    auto n1SrcShape = srcLayout.template GetShapeDim<1, srcExpectSize>();
    auto n2SrcShape = srcLayout.template GetShapeDim<2, srcExpectSize>();
    auto n3SrcShape = srcLayout.template GetShapeDim<3, srcExpectSize>();

    const auto idxLayout = idx.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<0, idxExpectSize>();

    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<index0, dstExpectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<index1, dstExpectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<index2, dstExpectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<index3, dstExpectSize>();
    auto n4DstStride = dstLayout.template GetStrideDim<index4, dstExpectSize>();

    auto n0DstShape = dstLayout.template GetShapeDim<index0, dstExpectSize>();
    auto n1DstShape = dstLayout.template GetShapeDim<index1, dstExpectSize>();
    auto n2DstShape = dstLayout.template GetShapeDim<index2, dstExpectSize>();
    auto n3DstShape = dstLayout.template GetShapeDim<index3, dstExpectSize>();
    auto n4DstShape = dstLayout.template GetShapeDim<index4, dstExpectSize>();

    auto srcOffset = srcLayout.template GetGmOffset<C1, 5>(srcCoordinate);
    auto idxOffset = idxLayout.template GetGmOffset<C2, 5>(idxCoordinate);
    using idxType = typename T2::Type;
    using srcType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ srcType *srcAddr = (__gm__ srcType *)((uint64_t)(src.GetAddr()));
    __gm__ idxType *idxAddr = (__gm__ idxType *)((uint64_t)(idx.GetAddr()));
    srcAddr += srcOffset;
    idxAddr += idxOffset;
    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)((uint64_t)(dst.GetAddr()));
    constexpr auto tileH = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<srcType, ShapeDim5, StrideDim5>;
    using TileDefine = pto::Tile<pto::TileType::Vec, dstType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    if constexpr (axis == 0) {
        __gm__ idxType *idx0 = idxAddr;
        for (LoopVar i = 0; i < n0DstShape; i++) {
            __gm__ dstType *src0 = srcAddr;
            __ubuf__ dstType *dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) {
                __ubuf__ dstType *dst1 = dst0;
                uint64_t index = idx0[j];
                src0 = srcAddr + index * n0SrcStride;
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    TileDefine dstTile(n3DstShape, n4DstShape);
                    GlobalData srcGlobal(src0, pto::Shape(1, 1, 1, n3DstShape, n4DstShape),
                        pto::Stride(0, 0, 0, n2SrcStride, n3SrcStride));
                    pto::TASSIGN(dstTile, (uint64_t)dst1);
                    pto::TLOAD(dstTile, srcGlobal);
                    dst1 += n2DstStride;
                    src0 += n1SrcStride;
                }
                dst0 += n1DstStride;
            }
            dstAddr += n0DstStride;
            idx0 += n0IdxStride;
        }
    } else if constexpr (axis == 1) {
        for (LoopVar i = 0; i < n0DstShape; i++) { // a
            __gm__ dstType *src0 = srcAddr;
            __gm__ idxType *idx0 = idxAddr;
            __ubuf__ dstType *dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) { // e
                __ubuf__ dstType *dst1 = dst0;
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    uint64_t index = idx0[k];
                    src0 = srcAddr + index * n1SrcStride;
                    TileDefine dstTile(n3DstShape, n4DstShape);
                    GlobalData srcGlobal(src0, pto::Shape(1, 1, 1, n3DstShape, n4DstShape),
                        pto::Stride(0, 0, 0, n2SrcStride, n3SrcStride));
                    pto::TASSIGN(dstTile, (uint64_t)dst1);
                    pto::TLOAD(dstTile, srcGlobal);
                    dst1 += n2DstStride;
                }
                dst0 += n1DstStride;
                idx0 += n0IdxStride;
            }
            dstAddr += n0DstStride;
            srcAddr += n0SrcStride;
        }
    } else if constexpr (axis == 2) {
        for (LoopVar i = 0; i < n0DstShape; i++) {
            __gm__ dstType *src0 = srcAddr;
            __ubuf__ dstType *dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) { // b
                __gm__ idxType *idx0 = idxAddr;
                __gm__ dstType *src1 = src0;
                __ubuf__ dstType *dst1 = dst0;
                for (LoopVar k = 0; k < n2DstShape; k++) { // e
                    __ubuf__ dstType *dst2 = dst1;
                    for (LoopVar l = 0; l < n3DstShape; l++) { // f
                        uint64_t index = idx0[l];
                        src1 = src0 + index * n2SrcStride;
                        TileDefine dstTile(1, n4DstShape);
                        GlobalData srcGlobal(
                            src1, pto::Shape(1, 1, 1, 1, n4DstShape), pto::Stride(0, 0, 0, n2SrcStride, n3SrcStride));
                        pto::TASSIGN(dstTile, (uint64_t)dst2);
                        pto::TLOAD(dstTile, srcGlobal);
                        dst2 += n3DstStride;
                    }
                    dst1 += n2DstStride;
                    idx0 += n0IdxStride;
                }
                src0 += n1SrcStride;
                dst0 += n1DstStride;
            }
            srcAddr += n0SrcStride;
            dstAddr += n0DstStride;
        }
    } else if constexpr (axis == 3) {
        for (LoopVar i = 0; i < n0DstShape; i++) {
            __gm__ dstType *src0 = srcAddr;
            __ubuf__ dstType *dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) { // b
                __gm__ dstType *src1 = src0;
                __ubuf__ dstType *dst1 = dst0;
                for (LoopVar k = 0; k < n2DstShape; k++) { // c
                    __gm__ dstType *src2 = src1;
                    __ubuf__ dstType *dst2 = dst1;
                    __gm__ idxType *idx0 = idxAddr;
                    for (LoopVar l = 0; l < n3DstShape; l++) { // e
                        __ubuf__ dstType *dst3 = dst2;
                        for (LoopVar p = 0; p < n4DstShape; p++) {
                            uint64_t index = idx0[p];
                            src2 = src1 + index;
                            *dst3 = *src2;
                            dst3++;
                        }
                        dst2 += n3DstStride;
                        idx0 += n0IdxStride;
                    }
                    src1 += n2SrcStride;
                    dst1 += n2DstStride;
                }
                src0 += n1SrcStride;
                dst0 += n1DstStride;
            }
            srcAddr += n0SrcStride;
            dstAddr += n0DstStride;
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID7);
    }
}

#endif
