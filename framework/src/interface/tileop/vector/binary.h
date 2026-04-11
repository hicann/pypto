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
 * \file binary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY__H
#define TILEOP_TILE_OPERATOR_BINARY__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "binary_brcinline.h"

template <BinaryOp op, pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADD(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TSUB(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMUL(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIV<PrecisionType>(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAX(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMIN(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::BITWISEAND) {
        pto::TAND(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::BITWISEOR) {
        pto::TOR(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::EXPANDEXPDIF) {
        pto::TCOLEXPANDEXPDIF(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::MOD) {
        pto::TFMOD(dst, src0, src1);
        return;
    }
}

template <BinaryOp op, pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, BrcMode brcmode, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryBrcDispatch(T0 dst, T1 src0, T2 src1)
{
    if constexpr (brcmode == BrcMode::BRC_W) {
        BinaryRowExpandComputeImpl<op, PrecisionType, LastUse>(dst, src0, src1);
    } else if constexpr (brcmode == BrcMode::BRC_H) {
        BinaryColExpandComputeImpl<op, PrecisionType, LastUse>(dst, src0, src1);
    } else if constexpr (brcmode == BrcMode::BRC_W0_H1) {
        pto::TCOLEXPAND(dst, src1);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        BinaryRowExpandComputeImpl<op, PrecisionType, LastUse>(dst, src0, dst);
    } else if constexpr (brcmode == BrcMode::BRC_H0_W1) {
        pto::TCOLEXPAND(dst, src0);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        BinaryRowExpandComputeImpl<op, PrecisionType, LastUse>(dst, dst, src1);
    } else {
        BinaryComputeImpl<op, PrecisionType, LastUse>(dst, src0, src1);
    }
}

template <
    BinaryOp op, pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, TileOp::BroadcastOperand WBrcSide,
    TileOp::PenuBroadcastOperand HBrcSide, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1)
{
    auto info = ExtractLayoutInfo(dst, src0, src1);
    using Src0TileInfo = TensorTileInfo<T1>;
    using Src1TileInfo = TensorTileInfo<T2>;
    constexpr BrcMode brcmode = GetBrcMode<WBrcSide, HBrcSide>();
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0PtoTile = typename std::conditional<
            (Src0TileInfo::tileW == 1 && WBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND),
            PtoTile<T1, pto::BLayout::ColMajor, true>, PtoTile<T1, pto::BLayout::RowMajor, true>>::type;
        using Src1PtoTile = typename std::conditional<
            (Src1TileInfo::tileW == 1 && WBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND),
            PtoTile<T2, pto::BLayout::ColMajor, true>, PtoTile<T2, pto::BLayout::RowMajor, true>>::type;
        auto src0Tile = Src0PtoTile().Data();
        auto src1Tile = Src1PtoTile().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryBrcDispatch<op, PrecisionType, brcmode, LastUse>(dstTile, src0Tile, src1Tile);
        return;
    }

    if constexpr (brcmode == BrcMode::BRC_HW) {
        BinaryMixBrcCompute<op, PrecisionType, WBrcSide, Src0TileInfo, Src1TileInfo, LastUse>(dst, src0, src1, info);
        return;
    }

    using Src0PtoTile = typename std::conditional<
        (Src0TileInfo::tileW == 1 && WBrcSide == TileOp::BroadcastOperand::LEFT_OPERAND),
        PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;
    using Src1PtoTile = typename std::conditional<
        (Src1TileInfo::tileW == 1 && WBrcSide == TileOp::BroadcastOperand::RIGHT_OPERAND),
        PtoTile<T2, pto::BLayout::ColMajor>, PtoTile<T2>>::type;
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = Src0PtoTile(src0);
    auto src1Tile = Src1PtoTile(src1);
    for (LoopVar n0Index = 0; n0Index < info.shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < info.shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < info.shape2; ++n2Index) {
                auto dsttileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto src0tileOffsets = TileOffset(
                    Src0TileInfo::tile0 == 1 ? 0 : n0Index, Src0TileInfo::tile1 == 1 ? 0 : n1Index,
                    Src0TileInfo::tile2 == 1 ? 0 : n2Index);
                auto src1tileOffsets = TileOffset(
                    Src1TileInfo::tile0 == 1 ? 0 : n0Index, Src1TileInfo::tile1 == 1 ? 0 : n1Index,
                    Src1TileInfo::tile2 == 1 ? 0 : n2Index);
                dstTile.Assign(dst, dsttileOffsets);
                src0Tile.Assign(src0, src0tileOffsets);
                src1Tile.Assign(src1, src1tileOffsets);
                BinaryBrcDispatch<op, PrecisionType, brcmode, LastUse>(
                    dstTile.Data(), src0Tile.Data(), src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ADD TAdd
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::ADD, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUB TSub
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::SUB, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MUL TMul
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MUL, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIV TDiv
template <
    pto::DivAlgorithm PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse = LastUse3Dim<0, 0, 0>,
    TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::DIV, PrecisionType, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAX TMax
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MAX, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MIN TMin
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MIN, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEAND TBitwiseAnd
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TBitwiseAnd(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::BITWISEAND, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEOR TBitwiseOr
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TBitwiseOr(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::BITWISEOR, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

TILEOP int gcd(int a, int b)
{
    if (a < 0) {
        a = 0 - a;
    }
    if (b < 0) {
        b = 0 - b;
    }
    while (a % b != 0) {
        int c = a % b;
        a = b;
        b = c;
    }
    return b;
}

#define OP_TILE_OP_GCD TGcd
template <
    TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TGcd(T0 dst, T1 src0, T2 src1)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    auto src0Addr = (__ubuf__ typename T1::Type*)((uint64_t)(src0.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));

    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    for (LoopVar n = 0; n < shape0; n++) {
        for (LoopVar j = 0; j < shape1; j++) {
            for (LoopVar k = 0; k < shape2; k++) {
                for (LoopVar m = 0; m < shape3; m++) {
                    for (LoopVar i = 0; i < shape4; i++) {
                        int tmpStride = n * dstStride0 + j * dstStride1 + k * dstStride2 + m * dstStride3 + i;
                        dstAddr[tmpStride] = gcd(src0Addr[tmpStride], src1Addr[tmpStride]);
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#define OP_TILE_OP_Mod TMod
template <
    typename LastUse = LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2>
TILEOP void TMod(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MOD, pto::DivAlgorithm::DEFAULT, WBrcSide, HBrcSide, LastUse>(dst, src0, src1);
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void SwitchPowResult(const size_t offset, T0 dst, T1 src0, T2 src1, T3 tmp)
{
    float a = src0.GetValue(offset);
    float b = src1.GetValue(offset);
    if (b == 0) {
        dst.SetValue(offset, static_cast<float>(1));
        return;
    }
    if (a == 0) {
        if (b > 0) {
            dst.SetValue(offset, static_cast<float>(0));
            return;
        }
        dst.SetValue(offset, static_cast<float>(1) / static_cast<float>(0));
        return;
    }
    if (TileOp::IsInteger(b)) {
        if (static_cast<int>(b) % 2 != 0 && a < 0) {
            dst.SetValue(offset, -tmp.GetValue(offset));
        } else {
            dst.SetValue(offset, tmp.GetValue(offset));
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void SwitchPowResult(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    constexpr size_t rowStride = src0.RowStride;
    constexpr size_t colStride = src0.ColStride;
    for (size_t n0 = 0, validRow = src0.GetValidRow(); n0 < validRow; ++n0) {
        for (size_t n1 = 0, validCol = src0.GetValidCol(); n1 < validCol; ++n1) {
            auto offset = n0 * rowStride + n1 * colStride;
            SwitchPowResult(offset, dst, src0, src1, tmp);
        }
    }
}

TILEOP int QuickPow(int a, int b)
{
    if (b == 0) {
        return 1;
    }
    if (a == 0) {
        return 0;
    }
    if (a == 1) {
        return 1;
    }
    if (a == -1) {
        if ((b & 1) == 0) {
            return 1;
        }
        return -1;
    }
    if (b < 0) {
        return 0;
    }
    int result = 1;
    int tmp = a;
    while (b != 0) {
        if ((b & 1) != 0) {
            result *= tmp;
        }
        b /= 2;
        tmp *= tmp;
    }
    return result;
}

template <typename T0, typename T1, typename T2>
TILEOP void IntegerPow(T0 dst, T1 src0, T2 src1)
{
    constexpr size_t rowStride = src0.RowStride;
    constexpr size_t colStride = src0.ColStride;
    for (size_t n0 = 0, validRow = src0.GetValidRow(); n0 < validRow; ++n0) {
        for (size_t n1 = 0, validCol = src0.GetValidCol(); n1 < validCol; ++n1) {
            auto offset = n0 * rowStride + n1 * colStride;
            int a = src0.GetValue(offset);
            int b = src1.GetValue(offset);
            dst.SetValue(offset, QuickPow(a, b));
        }
    }
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void CalcPow(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (std::is_same_v<typename T1::DType, int32_t>) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        IntegerPow(dst, src0, src1);
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    } else {
        pto::TABS(tmp, src0);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TLOG(tmp, tmp);
        pto::TLOG(dst, src0);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TMUL(tmp, tmp, src1);
        pto::TMUL(dst, dst, src1);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        pto::TEXP(tmp, tmp);
        pto::TEXP(dst, dst);
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#endif
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        SwitchPowResult(dst, src0, src1, tmp);
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
}

template <BinaryOp op, TileOp::BroadcastOperand WBrcSide, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpComputeImpl(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (op == BinaryOp::BITWISEXOR) {
        pto::TXOR(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::POW) {
        CalcPow(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::REM) {
        pto::TREM(dst, src0, src1, tmp);
        return;
    }
}

template <
    BinaryOp op, TileOp::BroadcastOperand WBrcSide, TileOp::PenuBroadcastOperand HBrcSide, typename T0, typename T1,
    typename T2, typename T3>
TILEOP void BinaryTmpCompute(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (TileOp::IsConstContinous<T0, T1, T2, T3>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto src0Tile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        auto src1Tile = PtoTile<T2, pto::BLayout::RowMajor, true>().Data();
        auto tmpTile = PtoTile<T3, pto::BLayout::RowMajor, true>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        pto::TASSIGN(tmpTile, (uint64_t)tmp.GetAddr());
        BinaryTmpComputeImpl<op, WBrcSide>(dstTile, src0Tile, src1Tile, tmpTile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    auto src1Tile = PtoTile<T2>(src1);
    auto tmpTile = PtoTile<T3>(tmp);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                src1Tile.Assign(src1, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                BinaryTmpComputeImpl<op, WBrcSide>(dstTile.Data(), src0Tile.Data(), src1Tile.Data(), tmpTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXOR TBitwiseXor
template <
    TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2,
    typename T3>
TILEOP void TBitwiseXor(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BinaryTmpCompute<BinaryOp::BITWISEXOR, WBrcSide, HBrcSide>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_POW TPow
template <
    TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2,
    typename T3>
TILEOP void TPow(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    static_assert(std::is_same_v<typename T1::Type, float> || std::is_same_v<typename T1::Type, int32_t>);
    BinaryTmpCompute<BinaryOp::POW, WBrcSide, HBrcSide>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REM TRem
template <
    TileOp::BroadcastOperand WBrcSide = TileOp::BroadcastOperand::NONE,
    TileOp::PenuBroadcastOperand HBrcSide = TileOp::PenuBroadcastOperand::NONE, typename T0, typename T1, typename T2,
    typename T3>
TILEOP void TRemainder(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BinaryTmpCompute<BinaryOp::REM, WBrcSide, HBrcSide>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_FLOORDIV TFloorDiv
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TFloorDiv(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    static_assert(std::is_same_v<typename T1::Type, int32_t>);

    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    auto offset =
                        n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 + n3Index * dstStride3;
#ifdef __DAV_V220
                    using FloatTileDefine =
                        pto::Tile<pto::TileType::Vec, float, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using IntTileDefine =
                        pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;

                    FloatTileDefine tmp0Tile(1, dstShape4);
                    FloatTileDefine tmp1Tile(1, dstShape4);
                    IntTileDefine src0Tile(1, dstShape4);
                    IntTileDefine src1Tile(1, dstShape4);
                    IntTileDefine dstTile(1, dstShape4);

                    pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));

                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    pipe_barrier(PIPE_V);
                    pto::TDIV(tmp0Tile, tmp0Tile, tmp1Tile);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
                    pipe_barrier(PIPE_V);
#else
                    using DataTileDefine =
                        pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using MaskTileDefine =
                        pto::Tile<pto::TileType::Vec, uint8_t, 1, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

                    DataTileDefine src0Tile(1, dstShape4);
                    DataTileDefine src1Tile(1, dstShape4);
                    DataTileDefine dstTile(1, dstShape4);
                    DataTileDefine tmp0DataTile(1, dstShape4);
                    DataTileDefine tmp1DataTile(1, dstShape4);

                    MaskTileDefine tmp0MaskTile(1, dstShape4);
                    MaskTileDefine tmp1MaskTile(1, dstShape4);

                    pto::TASSIGN(tmp0DataTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1DataTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));

                    // Reuse the same tmp as packed mask storage
                    pto::TASSIGN(tmp0MaskTile, (uint64_t)(tmp.GetAddr()));
                    pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tileW * dstTypeSize));

                    pto::TCMPS(tmp0MaskTile, src0Tile, 0, CmpMode::LT);
                    pto::TCMPS(tmp1MaskTile, src1Tile, 0, CmpMode::LT);
                    pto::TXOR(tmp0MaskTile, tmp0MaskTile, tmp1MaskTile, dstTile); // packed mask of sign_differ
                    pto::TDIV(dstTile, src0Tile, src1Tile);                       // quot
                    pto::TMUL(tmp1DataTile, src1Tile, dstTile);
                    pto::TMULS(tmp1DataTile, tmp1DataTile, -1);
                    pto::TADD(src0Tile, tmp1DataTile, src0Tile); // rem

                    pto::TCMPS(tmp1MaskTile, src0Tile, 0, CmpMode::NE);
                    pto::TAND(tmp0MaskTile, tmp0MaskTile, tmp1MaskTile);
                    pto::TADDS(src0Tile, dstTile, -1);
                    pto::TSEL(dstTile, tmp0MaskTile, src0Tile, dstTile, tmp1DataTile);
#endif
                }
            }
        }
    }
}
#endif
