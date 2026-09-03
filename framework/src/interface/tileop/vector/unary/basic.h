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
 * \file basic.h
 * \brief Unary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY_BASIC_H
#define TILEOP_TILE_OPERATOR_VEC_UNARY_BASIC_H

#include "utils/sync.h"
#include "../pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cmath>
#include <limits>

constexpr int16_t BF16_EXPONENT_MASK = 0x7F80;
constexpr int16_t FP16_EXPONENT_MASK = 0x7C00;
constexpr int16_t PACKED_BOOL_TRUE = 0x0101;

template <typename DType>
TILEOP constexpr bool IsIntegralType()
{
    return std::is_same_v<DType, int32_t> || std::is_same_v<DType, uint32_t> || std::is_same_v<DType, int8_t> ||
           std::is_same_v<DType, uint8_t> || std::is_same_v<DType, int16_t> || std::is_same_v<DType, uint16_t> ||
           std::is_same_v<DType, int64_t> || std::is_same_v<DType, uint64_t>;
}

template <UnaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1>
TILEOP void UnaryComputeImpl(T0 dst, T1 src)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == UnaryOp::EXP) {
        PTO_WITH_LAST_USE(pto::TEXP<PrecisionType>(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::RSQRT) {
        PTO_WITH_LAST_USE(pto::TRSQRT(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::SQRT) {
        PTO_WITH_LAST_USE(pto::TSQRT<PrecisionType>(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::BRCB) {
        PTO_WITH_LAST_USE(pto::TROWEXPAND(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::ABS) {
        PTO_WITH_LAST_USE(pto::TABS(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::RECIPROCAL) {
        PTO_WITH_LAST_USE(pto::TRECIP<PrecisionType>(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::BITWISENOT) {
        PTO_WITH_LAST_USE(pto::TNOT(dst, src), n1, n2);
    }
    if constexpr (op == UnaryOp::RELU) {
        pto::TMAXS(dst, src, static_cast<typename T1::DType>(0));
    }
    if constexpr (op == UnaryOp::LN) {
        pto::TLOG<PrecisionType>(dst, src);
    }
}

template <typename T, typename HalfTileDefineSrc, typename TileDefineDst, typename B16TileDefineSrc>
TILEOP void IsFiniteCalcImpl(TileDefineDst dst, B16TileDefineSrc src, B16TileDefineSrc bufferB16,
                             HalfTileDefineSrc bufferFP16)
{
    int16_t mask = 0;
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        mask = BF16_EXPONENT_MASK;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
        mask = FP16_EXPONENT_MASK;
    }
    pto::TANDS(bufferB16, src, mask);
    SyncV();
    pto::TSUBS(bufferB16, bufferB16, mask);
    SyncV();
    pto::TMAXS(bufferB16, bufferB16, (int16_t)-1);
    SyncV();
    pto::TMULS(bufferB16, bufferB16, (int16_t)-1);
    SyncV();
    pto::TCVT(dst, bufferFP16, pto::RoundMode::CAST_CEIL);
    SyncV();
}

template <typename T, typename HalfTileDefineSrc, bool CombineAxis, typename TileDefineDst, typename B16TileDefineSrc>
TILEOP void IsFiniteComputeImpl(TileDefineDst dst, B16TileDefineSrc src, HalfTileDefineSrc buffer)
{
    if constexpr (!CombineAxis) {
        HalfTileDefineSrc bufferFP16(src.GetValidRow(), src.GetValidCol());
        pto::TASSIGN(bufferFP16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        B16TileDefineSrc bufferB16(src.GetValidRow(), src.GetValidCol());
        pto::TASSIGN(bufferB16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        IsFiniteCalcImpl<T>(dst, src, bufferB16, bufferFP16);
    } else {
        HalfTileDefineSrc bufferFP16;
        pto::TASSIGN(bufferFP16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        B16TileDefineSrc bufferB16;
        pto::TASSIGN(bufferB16, reinterpret_cast<std::uintptr_t>(buffer.data()));
        IsFiniteCalcImpl<T>(dst, src, bufferB16, bufferFP16);
    }
}

template <UnaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1>
TILEOP void UnaryCompute(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>() && !std::is_same_v<typename T0::Type, int64_t> &&
                  !std::is_same_v<typename T0::Type, uint64_t>) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        UnaryComputeImpl<op, PrecisionType, LastUse>(dstTile, srcTile);
        return;
    }

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                UnaryComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

#define OP_TILE_OP_EXP TExp
template <typename LastUse, typename T0, typename T1>
TILEOP void BrcbCompute(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor>;
    using SrcTileDefine = typename std::conditional<
        (srcTileW == 1), pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::ColMajor>,
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileW, srcTileH, pto::BLayout::ColMajor>>::type;

    SrcTileDefine srcTile;
    DstTileDefine dstTile;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dstTileOffsets = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcTileOffsets = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstTileOffsets * sizeof(typename T0::Type)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcTileOffsets * sizeof(typename T1::Type)));
                UnaryComputeImpl<UnaryOp::BRCB, 0, LastUse>(dstTile, srcTile);
            }
        }
    }
}

#define OP_TILE_OP_EXP TExp
template <auto PrecisionType = pto::ExpAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename T0,
          typename T1>
TILEOP void TExp(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::EXP, PrecisionType, LastUse>(dst, src);
}

#define OP_TILE_OP_RSQRT TRsqrt
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TRsqrt(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::RSQRT, 0, LastUse>(dst, src);
}

#define OP_TILE_OP_SQRT TSqrt
template <auto PrecisionType = pto::SqrtAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename T0,
          typename T1>
TILEOP void TSqrt(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::SQRT, PrecisionType, LastUse>(dst, src);
}

template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFiniteCombineAxis(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer)
{
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t,
                                       typename DstTileTensor::Type>;
    using SrcType = typename SrcTileTensor::Type;

    constexpr size_t tileSrcH = GetMergedAxisIfNeed<SrcTileTensor, true>();
    constexpr size_t tileSrcW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t tileDstH = GetMergedAxisIfNeed<DstTileTensor, true>();
    constexpr size_t tileDstW = TileOp::GetTensorTileShapeDim<DstTileTensor, DIM_5TH, MAX_DIMS>();

    constexpr int validH = GetValidHeight<SrcTileTensor, true>();
    constexpr int validW = GetValidWidth<SrcTileTensor>();

    if constexpr (IsIntegralType<SrcType>()) {
        constexpr auto INT16_PACK_RATIO = sizeof(int16_t) / sizeof(uint8_t);
        using TileDefineDst = pto::Tile<pto::TileType::Vec, int16_t, tileDstH,
                                        (tileDstW + INT16_PACK_RATIO - 1) / INT16_PACK_RATIO, pto::BLayout::RowMajor,
                                        validH, (validW + INT16_PACK_RATIO - 1) / INT16_PACK_RATIO>;
        TileDefineDst dstTile;
        pto::TASSIGN(dstTile, dst.GetAddr());
        int16_t mask = PACKED_BOOL_TRUE;
        TANDS(dstTile, dstTile, 0);
        SyncV();
        TORS(dstTile, dstTile, mask);
        SyncV();
        return;
    } else {
        using TileDefineDst = pto::Tile<pto::TileType::Vec, DstType, tileDstH, tileDstW, pto::BLayout::RowMajor, validH,
                                        validW>;
        using HalfTileDefineSrc = pto::Tile<pto::TileType::Vec, half, tileSrcH,
                                            tileSrcW * sizeof(SrcType) / sizeof(half), pto::BLayout::RowMajor, validH,
                                            validW>;
        using B16TileDefineSrc = pto::Tile<pto::TileType::Vec, int16_t, tileSrcH,
                                           tileSrcW * sizeof(SrcType) / sizeof(int16_t), pto::BLayout::RowMajor, validH,
                                           validW>;

        HalfTileDefineSrc bufferTile;
        TileDefineDst dstTile;
        B16TileDefineSrc srcTile;
        pto::TASSIGN(bufferTile, buffer.GetAddr());
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());

        if constexpr (std::is_same_v<SrcType, float>) {
            using FP32TileDefineSrc = pto::Tile<pto::TileType::Vec, float, tileSrcH, tileSrcW, pto::BLayout::RowMajor,
                                                validH, validW>;
            FP32TileDefineSrc srcFP32;
            HalfTileDefineSrc srcFP16;
            pto::TASSIGN(srcFP32, src.GetAddr());
            pto::TASSIGN(srcFP16, src.GetAddr());
            pto::TCVT(srcFP16, srcFP32, pto::RoundMode::CAST_NONE);
            SyncV();
        }

        IsFiniteComputeImpl<SrcType, HalfTileDefineSrc, true>(dstTile, srcTile, bufferTile);
    }
}

template <typename DstTileTensor, typename SrcTileTensor>
TILEOP void TIsFinite4Integral(DstTileTensor dst, SrcTileTensor src)
{
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t,
                                       typename DstTileTensor::Type>;
    using SrcType = typename SrcTileTensor::Type;
    constexpr size_t tileSrcH = GetMergedAxisIfNeed<SrcTileTensor, false>();
    constexpr size_t tileSrcW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t tileDstH = GetMergedAxisIfNeed<DstTileTensor, false>();
    constexpr size_t tileDstW = TileOp::GetTensorTileShapeDim<DstTileTensor, DIM_5TH, MAX_DIMS>();

    static_assert(IsElementwiseDstLayoutCoveredByOperand<DstTileTensor, SrcTileTensor>());
    int validH = dst.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>();
    int validW = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto INT16_PACK_RATIO = sizeof(int16_t) / sizeof(uint8_t);
    using TileDefineDst = pto::Tile<pto::TileType::Vec, int16_t, tileDstH, tileDstW / INT16_PACK_RATIO,
                                    pto::BLayout::RowMajor, -1, -1>;
    TileDefineDst dstTile(validH, (validW + INT16_PACK_RATIO - 1) / INT16_PACK_RATIO);
    pto::TASSIGN(dstTile, dst.GetAddr());
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    int16_t mask = PACKED_BOOL_TRUE;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(DstType));
                TANDS(dstTile, dstTile, 0);
                SyncV();
                TORS(dstTile, dstTile, mask);
                SyncV();
            }
        }
    }
}

template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFinite4Floats(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer)
{
    using SrcType = typename SrcTileTensor::Type;
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t,
                                       typename DstTileTensor::Type>;
    static_assert(IsElementwiseDstLayoutCoveredByOperand<DstTileTensor, SrcTileTensor>());
    constexpr size_t tileSrcH = ElementwiseOperandExecConfig<DstTileTensor, SrcTileTensor>::tileH;
    constexpr size_t tileSrcW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    constexpr size_t tileDstH = GetMergedAxisIfNeed<DstTileTensor, false>();
    constexpr size_t tileDstW = TileOp::GetTensorTileShapeDim<DstTileTensor, DIM_5TH, MAX_DIMS>();

    int validH = dst.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>();
    int validW = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using TileDefineDst = pto::Tile<pto::TileType::Vec, DstType, tileDstH, tileDstW, pto::BLayout::RowMajor, -1, -1>;
    using HalfTileDefineSrc = pto::Tile<pto::TileType::Vec, half, tileSrcH, tileSrcW * sizeof(SrcType) / sizeof(half),
                                        pto::BLayout::RowMajor, -1, -1>;
    using B16TileDefineSrc = pto::Tile<pto::TileType::Vec, int16_t, tileSrcH,
                                       tileSrcW * sizeof(SrcType) / sizeof(int16_t), pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefineSrc bufferTile(validH, validW);
    pto::TASSIGN(bufferTile, buffer.GetAddr());

    TileDefineDst dstTile(validH, validW);
    B16TileDefineSrc srcTile(validH, validW);

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(DstType));
                pto::TASSIGN(srcTile, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(int16_t));
                if constexpr (std::is_same_v<SrcType, float>) {
                    using FP32TileDefineSrc = pto::Tile<pto::TileType::Vec, float, tileSrcH, tileSrcW,
                                                        pto::BLayout::RowMajor, -1, -1>;
                    FP32TileDefineSrc srcFP32(validH, validW);
                    HalfTileDefineSrc srcFP16(validH, validW);
                    pto::TASSIGN(srcFP32, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(float));
                    pto::TASSIGN(srcFP16, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(half));
                    pto::TCVT(srcFP16, srcFP32, pto::RoundMode::CAST_NONE);
                    SyncV();
                }
                IsFiniteComputeImpl<SrcType, HalfTileDefineSrc, false>(dstTile, srcTile, bufferTile);
            }
        }
    }
}

#define OP_TILE_OP_ISFINITE TIsFinite
template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFinite(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer)
{
    constexpr bool sameExecSize = GetAllAxisTileProduct<DstTileTensor>() == GetAllAxisTileProduct<SrcTileTensor>();
    if constexpr (TileOp::IsConstContinous<DstTileTensor, SrcTileTensor>() && sameExecSize) {
        TIsFiniteCombineAxis(dst, src, buffer);
        return;
    }

    using SrcType = typename SrcTileTensor::Type;
    if constexpr (IsIntegralType<SrcType>()) {
        TIsFinite4Integral(dst, src);
    } else {
        TIsFinite4Floats(dst, src, buffer);
    }
}

#define OP_TILE_OP_BRCB Tbrcb
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void Tbrcb(T0 dst, T1 src)
{
    BrcbCompute<LastUse>(dst, src);
}

#define OP_TILE_OP_ABS TAbs
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TAbs(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::ABS, 0, LastUse>(dst, src);
}

#define OP_TILE_OP_BITWISENOT TBitwiseNot
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TBitwiseNot(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::BITWISENOT, 0, LastUse>(dst, src);
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void CeilComputeImpl(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::DType>) {
        pto::TMOV(dst, src);
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_CEIL);
    }
}
#define OP_TILE_OP_CEIL TCEIL
template <typename T0, typename T1>
TILEOP void TCeil(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr()) {
            return;
        }
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        CeilComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                CeilComputeImpl<float>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void FloorComputeImpl(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::DType>) {
        pto::TMOV(dst, src);
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_FLOOR);
    }
}
#define OP_TILE_OP_FLOOR TFLOOR
template <typename T0, typename T1>
TILEOP void TFloor(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr()) {
            return;
        }
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        FloorComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                FloorComputeImpl<float>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void TruncComputeImpl(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::DType>) {
        pto::TMOV(dst, src);
    } else {
        pto::TCVT(dst, src, pto::RoundMode::CAST_TRUNC);
    }
}
#define OP_TILE_OP_TRUNC TTRUNC
template <typename T0, typename T1>
TILEOP void TTrunc(T0 dst, T1 src)
{
    if constexpr (std::is_integral_v<typename T1::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr()) {
            return;
        }
    }

    using SrcExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
    if constexpr (TileOp::IsConstContinous<T0, T1>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T0, pto::BLayout::RowMajor, true, SrcExecDtype>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        TruncComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                TruncComputeImpl<float>(dstTile.Data(), srcExecTile);
            }
        }
    }
}

#define OP_TILE_OP_ROUND TRound
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TRound(T0 dst, T1 tmp, T2 src, Scalar powDecimals)
{
    if constexpr (std::is_integral_v<typename T2::Type>) {
        if ((uint64_t)dst.GetAddr() == (uint64_t)src.GetAddr() && powDecimals >= 1.0f) {
            return;
        }
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                if constexpr (std::is_integral_v<typename T2::Type>) {
                    if (powDecimals >= 1.0f) {
                        if constexpr (std::is_same_v<typename T2::Type, int64_t>) {
                            pto::TADDS(dstTile.Data(), srcExecTile, static_cast<typename T2::Type>(0));
                        } else {
                            pto::TMOV(dstTile.Data(), srcExecTile);
                        }
                        continue;
                    }
                }

                if constexpr (std::is_same_v<typename T2::Type, float>) {
                    pto::TMULS(srcExecTile, srcExecTile, powDecimals);
                    SyncV();
                    pto::TCVT(srcExecTile, srcExecTile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TDIVS(dstTile.Data(), srcExecTile, powDecimals);
                } else {
                    pto::TCVT(tmpTile.Data(), srcExecTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TMULS(tmpTile.Data(), tmpTile.Data(), powDecimals);
                    SyncV();
                    pto::TCVT(tmpTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TMULS(tmpTile.Data(), tmpTile.Data(), 1.0f / powDecimals);
                    SyncV();
                    pto::TCVT(dstTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
                }
            }
        }
    }
}

#define OP_TILE_OP_RECIPROCAL TReciprocal
template <auto PrecisionType = pto::RecipAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename T0,
          typename T1>
TILEOP void TReciprocal(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::RECIPROCAL, PrecisionType, LastUse>(dst, src);
}

#define OP_TILE_OP_RELU TRelu
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TRelu(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::RELU, 0, LastUse>(dst, src);
}

template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void PackCompute(T0 dst, T1 src)
{
    auto shape = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    if constexpr (std::is_same_v<typename T0::Type, int64_t> || std::is_same_v<typename T0::Type, uint64_t>) {
        constexpr auto INT64_TO_INT32_RATIO = sizeof(int64_t) / sizeof(int32_t);
        using TileDef = pto::Tile<pto::TileType::Vec, int32_t, 1, tileW * INT64_TO_INT32_RATIO, pto::BLayout::RowMajor,
                                  -1, -1>;
        TileDef dstTile(1, shape * INT64_TO_INT32_RATIO);
        TileDef srcTile(1, shape * INT64_TO_INT32_RATIO);
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());
        pto::TMOV(dstTile, srcTile);
    } else {
        using TileDef = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileDef dstTile(1, shape);
        TileDef srcTile(1, shape);
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());
        pto::TMOV(dstTile, srcTile);
    }
}

#define OP_TILE_OP_PACK TPack
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TPack(T0 dst, T1 src)
{
    PackCompute(dst, src);
}

#define OP_TILE_OP_UNPACK TUnPack
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1>
TILEOP void TUnPack(T0 dst, T1 src)
{
    PackCompute(dst, src);
}

#endif // TILEOP_TILE_OPERATOR_VEC_UNARY_BASIC_H
