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
#include "tileop/utils/type_traits.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "binary_brcinline.h"

template <BinaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename T2>
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
        pto::TFMOD<PrecisionType>(dst, src0, src1);
        return;
    }
}

template <BinaryOp op, auto PrecisionType = 0, BrcMode brcmode, typename LastUse, typename T0, typename T1, typename T2>
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
    } else if constexpr (brcmode == BrcMode::NONE) {
        BinaryComputeImpl<op, PrecisionType, LastUse>(dst, src0, src1);
    }
}

template <BrcMode brcmode, typename Src0Tensor, typename Src1Tensor, typename Src0TileInfo, typename Src1TileInfo,
          int... BrcOperands, typename T1, typename T2>
TILEOP void A5Expand1DimBrcWSrc(T1 src0, T2 src1, uint64_t src0Addr, uint64_t src1Addr)
{
    constexpr bool src0NeedExpand = brcmode == BrcMode::BRC_W &&
                                    GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT &&
                                    Std::tuple_size<typename Src0Tensor::Shape>::value == 1 && Src0TileInfo::tileW != 1;
    constexpr bool src1NeedExpand = brcmode == BrcMode::BRC_W &&
                                    GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT &&
                                    Std::tuple_size<typename Src1Tensor::Shape>::value == 1 && Src1TileInfo::tileW != 1;
    if constexpr (src0NeedExpand) {
        using FillDst = pto::Tile<pto::TileType::Vec, typename T1::DType, T1::Rows, T1::Cols, pto::BLayout::RowMajor,
                                  T1::Rows, T1::Cols>;
        FillDst fillDst;
        pto::TASSIGN(fillDst, src0Addr);
        pto::TROWEXPAND(fillDst, src0);
    } else if constexpr (src1NeedExpand) {
        using FillDst = pto::Tile<pto::TileType::Vec, typename T2::DType, T2::Rows, T2::Cols, pto::BLayout::RowMajor,
                                  T2::Rows, T2::Cols>;
        FillDst fillDst;
        pto::TASSIGN(fillDst, src1Addr);
        pto::TROWEXPAND(fillDst, src1);
    }
}

template <BinaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryElementwiseCompute(T0 dst, T1 src0, T2 src1)
{
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0ExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
        using Src1ExecDtype = typename ElementwiseOperandExecConfig<T0, T2>::OperandDtype;
        using Src0TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src0ExecDtype>::Type;
        using Src1TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src1ExecDtype>::Type;
        Src0TileDefine src0ExecTile;
        Src1TileDefine src1ExecTile;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0ExecTile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1ExecTile, (uint64_t)src1.GetAddr());
        BinaryComputeImpl<op, PrecisionType, LastUse>(dstTile, src0ExecTile, src1ExecTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                BinaryComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), src0ExecTile, src1ExecTile);
            }
        }
    }
}

template <BinaryOp op, auto PrecisionType = 0, typename LastUse, int... BrcOperands, typename T0, typename T1,
          typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }
    using Src0TileInfo = TensorTileInfo<T1>;
    using Src1TileInfo = TensorTileInfo<T2>;
    constexpr BrcMode brcmode = GetBrcMode<BrcOperands...>();
    if constexpr (!TileOp::HasBrcOperand<BrcOperands...>() && op != BinaryOp::EXPANDEXPDIF) {
        BinaryElementwiseCompute<op, PrecisionType, LastUse>(dst, src0, src1);
        return;
    } else if constexpr (brcmode == BrcMode::BRC_HW) {
        BinaryMixBrcCompute<op, PrecisionType, Src0TileInfo, Src1TileInfo, LastUse, BrcOperands...>(dst, src0, src1);
        return;
    } else if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true && !TileOp::HasBrcOperand<BrcOperands...>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0PtoTile = typename std::conditional<
            (Src0TileInfo::tileW == 1 && GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT),
            PtoTile<T1, pto::BLayout::ColMajor, true>, PtoTile<T1, pto::BLayout::RowMajor, true>>::type;
        using Src1PtoTile = typename std::conditional<
            (Src1TileInfo::tileW == 1 && GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT),
            PtoTile<T2, pto::BLayout::ColMajor, true>, PtoTile<T2, pto::BLayout::RowMajor, true>>::type;
        auto src0Tile = Src0PtoTile().Data();
        auto src1Tile = Src1PtoTile().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryBrcDispatch<op, PrecisionType, brcmode, LastUse>(dstTile, src0Tile, src1Tile);
        return;
    }

    using Src0PtoTile = typename std::conditional<(Src0TileInfo::tileW == 1 &&
                                                   GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT),
                                                  PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;
    using Src1PtoTile = typename std::conditional<(Src1TileInfo::tileW == 1 &&
                                                   GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT),
                                                  PtoTile<T2, pto::BLayout::ColMajor>, PtoTile<T2>>::type;
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = Src0PtoTile(src0);
    auto src1Tile = Src1PtoTile(src1);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dsttileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto src0tileOffsets = TileOffset(
                    (Src0TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_LEFT) ? 0 : n0Index,
                    (Src0TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_LEFT) ? 0 : n1Index,
                    (Src0TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_LEFT) ? 0 : n2Index);
                auto src1tileOffsets = TileOffset(
                    (Src1TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : n0Index,
                    (Src1TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : n1Index,
                    (Src1TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 :
                                                                                                            n2Index);
                dstTile.Assign(dst, dsttileOffsets);
                src0Tile.Assign(src0, src0tileOffsets);
                src1Tile.Assign(src1, src1tileOffsets);
#if defined PTO_NPU_ARCH_A5 || defined(__LITE_NPU)
                if constexpr (GetBrcOperandAt<DIM_5TH, BrcOperands...>() != BRC_NONE) {
                    A5Expand1DimBrcWSrc<brcmode, T1, T2, Src0TileInfo, Src1TileInfo, BrcOperands...>(
                        src0Tile.Data(), src1Tile.Data(), (uint64_t)src0.GetAddr(), (uint64_t)src1.GetAddr());
                }
#endif
                BinaryBrcDispatch<op, PrecisionType, brcmode, LastUse>(dstTile.Data(), src0Tile.Data(),
                                                                       src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ADD TAdd
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::ADD, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_SUB TSub
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::SUB, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_MUL TMul
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MUL, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_DIV TDiv
template <auto PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands,
          typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::DIV, PrecisionType, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_MAX TMax
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MAX, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_MIN TMin
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MIN, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEAND TBitwiseAnd
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TBitwiseAnd(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::BITWISEAND, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEOR TBitwiseOr
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TBitwiseOr(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::BITWISEOR, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_GCD TGcd
template <int... BrcOperands, typename T0, typename T1, typename T2, typename T3>
TILEOP void TGcd(T0 dst, T1 src0, T2 src1, T3 temp)
{
    // 计算位宽按输入类型选择：int8/uint8 输入时中间值 <= 510（q*b <= a+b），int16 槽不溢出；
    // int16/int32 输入时 q*b 可达 a+b <= 65534/2e8，需 int32 槽。
    using CalcType = typename std::conditional<Std::is_same_v<typename T0::Type, int8_t> ||
                                                   Std::is_same_v<typename T0::Type, uint8_t>,
                                               int16_t, int32_t>::type;
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    using CalcTile = pto::Tile<pto::TileType::Vec, CalcType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TTile = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using FTile = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using F16Tile = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    // TCMPS/TCMP/TSEL 的掩码按位打包存放：每行 ceil(tileW/32)*4 字节，按 32B 对齐
    constexpr auto maskCols = ((tileW + 7) / 8 + 31) / 32 * 32;
    using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, maskCols, pto::BLayout::RowMajor, -1, -1>;

    const auto dstLayout = dst.GetLayout();
    const auto src0Layout = src0.GetLayout();
    const auto src1Layout = src1.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }
    using Src0TileInfo = TensorTileInfo<T1>;
    using Src1TileInfo = TensorTileInfo<T2>;
    constexpr BrcMode brcmode = GetBrcMode<BrcOperands...>();

    // temp 布局（与 BinaryOperationTileFunc 中 GCD 的临时空间分配保持一致）：
    //   slot0: a = |src0|  slot1: b = |src1|  slot2: r = a % b  slot3: q
    //   slot4: 暂存，顺序复用为 q*b / q修正 / r修正（生命周期互不重叠）
    //   f/fb: 两个 f32 槽（近似商与 refine 除法）
    //   之后依次为：冻结掩码、修正掩码、TSEL tmp（256B+64B 保护带）
    // 计算槽按 CalcType 位宽（int8/uint8 输入为 int16），f32 槽与掩码固定大小
    constexpr auto calcStride = tileH * tileW * sizeof(CalcType);
    constexpr auto floatBytes = tileH * tileW * sizeof(float);
    constexpr auto maskBytes = tileH * maskCols;
    constexpr auto fBase = 6 * calcStride;
    constexpr auto fbBase = fBase + floatBytes;
    constexpr auto maskBase = fbBase + floatBytes;
    constexpr auto bNegBase = maskBase + maskBytes;
    constexpr auto oppBase = bNegBase + maskBytes;
    constexpr auto fixMaskBase = oppBase + maskBytes;
    constexpr auto selTmpBase = fixMaskBase + maskBytes;

    CalcTile aTile(shape3, shape4);
    CalcTile bTile(shape3, shape4);
    CalcTile rTile(shape3, shape4);
    CalcTile qTile(shape3, shape4);
    CalcTile scratchTile(shape3, shape4);
    CalcTile auxTile(shape3, shape4);
    FTile fTile(shape3, shape4);
    FTile fbTile(shape3, shape4);
    MaskTile maskTile(shape3, shape4);
    MaskTile bNegTile(shape3, shape4);
    MaskTile oppTile(shape3, shape4);
    MaskTile fixMaskTile(shape3, shape4);
    CalcTile selTmpTile(shape3, shape4);
    TTile src0Tile(shape3, shape4);
    TTile src1Tile(shape3, shape4);
    TTile dstTile(shape3, shape4);

    TASSIGN(aTile, temp.GetAddr());
    TASSIGN(bTile, temp.GetAddr() + calcStride);
    TASSIGN(rTile, temp.GetAddr() + 2 * calcStride);
    TASSIGN(qTile, temp.GetAddr() + 3 * calcStride);
    TASSIGN(scratchTile, temp.GetAddr() + 4 * calcStride);
    TASSIGN(auxTile, temp.GetAddr() + 5 * calcStride);
    TASSIGN(fTile, temp.GetAddr() + fBase);
    TASSIGN(fbTile, temp.GetAddr() + fbBase);
    TASSIGN(maskTile, temp.GetAddr() + maskBase);
    TASSIGN(bNegTile, temp.GetAddr() + bNegBase);
    TASSIGN(oppTile, temp.GetAddr() + oppBase);
    TASSIGN(fixMaskTile, temp.GetAddr() + fixMaskBase);
    TASSIGN(selTmpTile, temp.GetAddr() + selTmpBase);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto src0tileOffsets = TileOffset(
                    (Src0TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_LEFT) ? 0 : n0Index,
                    (Src0TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_LEFT) ? 0 : n1Index,
                    (Src0TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_LEFT) ? 0 : n2Index);
                auto src1tileOffsets = TileOffset(
                    (Src1TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : n0Index,
                    (Src1TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : n1Index,
                    (Src1TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 :
                                                                                                            n2Index);
                auto src0Offset = GenTileOffset(src0, src0tileOffsets);
                auto src1Offset = GenTileOffset(src1, src1tileOffsets);
                TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(typename T0::Type)));
                TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * sizeof(typename T1::Type)));
                TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * sizeof(typename T2::Type)));

                // 统一转换到计算位宽。a2/a3 的 TCVT 没有 int8/uint8/int16 到 int32 的
                // 直接转换，int16 经 f32、int8/uint8 经 f16 中转（小整数转换无精度损失）
                if constexpr (Std::is_same_v<typename T0::Type, int32_t>) {
                    pto::TMOV(aTile, src0Tile);
                    pto::TMOV(bTile, src1Tile);
                } else if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    pto::TCVT(fTile, src0Tile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(fbTile, src1Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(aTile, fTile, pto::RoundMode::CAST_RINT);
                    pto::TCVT(bTile, fbTile, pto::RoundMode::CAST_RINT);
                } else {
                    // int8/uint8: s8/u8 -> f16 -> s16（f16 中转视图复用 fbTile 的地址）
                    F16Tile f16Tile(shape3, shape4);
                    TASSIGN(f16Tile, temp.GetAddr() + fbBase);
                    pto::TCVT(f16Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(aTile, f16Tile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TCVT(f16Tile, src1Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(bTile, f16Tile, pto::RoundMode::CAST_RINT);
                }
                SyncV();

                // |x| = x * clamp(x, -1, 1)
                pto::TMINS(rTile, aTile, 1);
                SyncV();
                pto::TMAXS(rTile, rTile, -1);
                SyncV();
                pto::TMUL(aTile, aTile, rTile);
                SyncV();
                pto::TMINS(rTile, bTile, 1);
                SyncV();
                pto::TMAXS(rTile, rTile, -1);
                SyncV();
                pto::TMUL(bTile, bTile, rTile);
                SyncV();
                if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    // int16 回绕: -32768 的绝对值 32768 回绕为 -32768（与 torch.gcd 一致）
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(maskTile, fTile, 32768.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, aTile, -65536);
                    SyncV();
                    pto::TSEL(aTile, maskTile, scratchTile, aTile, selTmpTile);
                    SyncV();
                    pto::TCVT(fTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(maskTile, fTile, 32768.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, bTile, -65536);
                    SyncV();
                    pto::TSEL(bTile, maskTile, scratchTile, bTile, selTmpTile);
                    SyncV();
                } else if constexpr (Std::is_same_v<typename T0::Type, int8_t>) {
                    // int8 回绕: -128 的绝对值 128 回绕为 -128（与 torch.gcd 一致）
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(maskTile, fTile, 128.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, aTile, -256);
                    SyncV();
                    pto::TSEL(aTile, maskTile, scratchTile, aTile, selTmpTile);
                    SyncV();
                    pto::TCVT(fTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(maskTile, fTile, 128.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, bTile, -256);
                    SyncV();
                    pto::TSEL(bTile, maskTile, scratchTile, bTile, selTmpTile);
                    SyncV();
                }

                // Euclid 迭代：固定迭代次数，避免仿真器上 while + GetValue 的标量同步竞态（卡死）。
                // b == 0 的 lane 用掩码冻结，收敛后保持 (gcd, 0) 不变；除零时 vdiv 结果
                // 为 inf/0，商饱和后与 0 相乘仍得 0，余数保持 a，冻结逻辑不受影响。
                // 最大迭代次数：int32 <= 1e8 时 Euclid 步数 <= 40（Fibonacci 界），64 封顶；
                // int16 <= 32767 时 <= 23，32 封顶；int8/uint8 <= 255 时 <= 13，16 封顶。
                constexpr int kMaxIter = Std::is_same_v<typename T0::Type, int32_t> ?
                                             64 :
                                             (Std::is_same_v<typename T0::Type, int16_t> ? 32 : 16);
                for (LoopVar k = 0; k < kMaxIter; ++k) {
                    // 冻结 b == 0 的 lane，避免除零
                    pto::TCMPS(maskTile, bTile, 0, pto::CmpMode::EQ);
                    SyncV();
                    // 符号归一化副本（a/b 原值不动，链更新用原始符号值）：
                    // aDiv = bNeg ? -a : a（aux）；bDiv = bNeg ? -b : b（rTile，bDiv > 0）
                    pto::TCVT(fbTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(bNegTile, fbTile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TMULS(auxTile, aTile, -1);
                    SyncV();
                    pto::TSEL(auxTile, bNegTile, auxTile, aTile, selTmpTile);
                    SyncV();
                    pto::TMULS(rTile, bTile, -1);
                    SyncV();
                    pto::TSEL(rTile, bNegTile, rTile, bTile, selTmpTile);
                    SyncV();
                    // floored 精确余数（bDiv > 0，aDiv 任意符号，结果 r_f = aux ∈ [0, bDiv)）
                    pto::TCVT(fbTile, rTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(fTile, fTile, fbTile);
                    SyncV();
                    pto::TCVT(qTile, fTile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                    pto::TMUL(scratchTile, qTile, rTile);
                    SyncV();
                    pto::TSUB(auxTile, auxTile, scratchTile); // aux = r = aDiv - q*bDiv
                    SyncV();
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(fTile, fTile, fbTile);
                    SyncV();
                    pto::TCVT(scratchTile, fTile, pto::RoundMode::CAST_FLOOR); // scratch = qc
                    SyncV();
                    // r_new = r_old - qc * bDiv（aux 已是 r_old，不能从 aDiv 重算）
                    pto::TMUL(scratchTile, scratchTile, rTile);
                    SyncV();
                    pto::TSUB(auxTile, auxTile, scratchTile);
                    SyncV();
                    // floored +/-1 修正（f32 符号桥接，bDiv > 0）
                    pto::TSUB(scratchTile, auxTile, rTile);
                    SyncV();
                    pto::TCVT(fTile, scratchTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(fixMaskTile, fTile, 0.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TSEL(auxTile, fixMaskTile, scratchTile, auxTile, selTmpTile);
                    SyncV();
                    pto::TADD(scratchTile, auxTile, rTile);
                    SyncV();
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(fixMaskTile, fTile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TSEL(auxTile, fixMaskTile, scratchTile, auxTile, selTmpTile);
                    SyncV();
                    // r_f == 0 标记（守卫用）
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(fixMaskTile, fTile, 0.0f, pto::CmpMode::EQ);
                    SyncV();
                    // 转换为 C 截断余数（用原始 a/b 的符号）：
                    //   opp = (f32(a) * f32(b) < 0)（原始异号，乘积符号精确）
                    //   r_trunc = opp ? (bDiv - r_f) : (bNeg ? -r_f : r_f)
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(fbTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TMUL(fTile, fTile, fbTile);
                    SyncV();
                    pto::TCMPS(oppTile, fTile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TMULS(qTile, auxTile, -1); // qTile = -r_f（q 已死）
                    SyncV();
                    pto::TSUB(scratchTile, rTile, auxTile); // bDiv - r_f（bNeg 分支）
                    SyncV();
                    pto::TSUB(rTile, auxTile, rTile); // r_f - bDiv（非 bNeg 分支, bDiv 已死）
                    SyncV();
                    pto::TSEL(scratchTile, bNegTile, scratchTile, rTile, selTmpTile); // rOpp
                    SyncV();
                    pto::TSEL(qTile, bNegTile, qTile, auxTile, selTmpTile); // rBase
                    SyncV();
                    pto::TSEL(auxTile, oppTile, scratchTile, qTile, selTmpTile); // r = opp ? rOpp : rBase
                    SyncV();
                    // r_f == 0 时保持 0（±b 毛刺会翻转最终符号; 零 tile 用自减生成）
                    pto::TSUB(scratchTile, auxTile, auxTile); // 0
                    SyncV();
                    pto::TSEL(scratchTile, fixMaskTile, scratchTile, auxTile, selTmpTile);
                    SyncV();
                    pto::TMOV(auxTile, scratchTile);
                    SyncV();
                    // 冻结 lane：a, b 保持不变（原始符号值），其余 a = b, b = a % b
                    pto::TSEL(aTile, maskTile, aTile, bTile, selTmpTile);
                    SyncV();
                    pto::TSEL(bTile, maskTile, bTile, auxTile, selTmpTile);
                    SyncV();
                } // 结果取非负 gcd（与 torch.gcd 语义一致）
                if constexpr (Std::is_same_v<typename T0::Type, int32_t>) {
                    pto::TMOV(dstTile, aTile);
                } else if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    pto::TCVT(dstTile, aTile, pto::RoundMode::CAST_RINT);
                } else {
                    // int8/uint8: s16 -> f32 -> f16 -> s8/u8（a2/a3 无直接转换）
                    F16Tile f16Tile(shape3, shape4);
                    TASSIGN(f16Tile, temp.GetAddr() + fbBase);
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TCVT(f16Tile, fTile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TCVT(dstTile, f16Tile, pto::RoundMode::CAST_RINT);
                }
                SyncV();
            }
        }
    }
}

#define OP_TILE_OP_Mod TMod
template <auto PrecisionType = pto::FmodAlgorithm::DEFAULT, typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands,
          typename T0, typename T1, typename T2>
TILEOP void TMod(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MOD, PrecisionType, LastUse, BrcOperands...>(dst, src0, src1);
}

template <BinaryOp op, auto PrecisionType = 0, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpComputeImpl(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (op == BinaryOp::BITWISEXOR) {
        pto::TXOR(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::POW) {
        pto::TPOW<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::REM) {
        pto::TREM<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryOp op, auto PrecisionType = 0, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpCompute(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (op != BinaryOp::REM && TileOp::IsConstContinous<T0, T1, T2, T3>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0ExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
        using Src1ExecDtype = typename ElementwiseOperandExecConfig<T0, T2>::OperandDtype;
        using TmpExecDtype = typename ElementwiseOperandExecConfig<T0, T3>::OperandDtype;
        using Src0TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src0ExecDtype>::Type;
        using Src1TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src1ExecDtype>::Type;
        using TmpTileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, TmpExecDtype>::Type;
        Src0TileDefine src0ExecTile;
        Src1TileDefine src1ExecTile;
        TmpTileDefine tmpExecTile;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0ExecTile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1ExecTile, (uint64_t)src1.GetAddr());
        pto::TASSIGN(tmpExecTile, (uint64_t)tmp.GetAddr());
        BinaryTmpComputeImpl<op, PrecisionType>(dstTile, src0ExecTile, src1ExecTile, tmpExecTile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    if constexpr (op == BinaryOp::REM) {
        auto tmpTile = PtoTile<T3>(tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                    tmpTile.Assign(tmp, tileOffsets);
                    BinaryTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1ExecTile, tmpTile.Data());
                }
            }
        }
    } else {
        auto tmpExecTile = MakeElementwiseOperandExecTile(dst, tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                    AssignElementwiseOperandExecTile(tmpExecTile, tmp, tileOffsets);
                    BinaryTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1ExecTile, tmpExecTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXOR TBitwiseXor
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitwiseXor(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BinaryTmpCompute<BinaryOp::BITWISEXOR>(dst, src0, src1, tmp);
}

#ifdef __DAV_V220
template <typename DstTile, typename Src0Tile, typename Src1Tile, typename TmpTile, typename MaskTile>
TILEOP void TPowFloatTile(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, TmpTile tmp0Tile, TmpTile tmp1Tile,
                          MaskTile mask0Tile, MaskTile mask1Tile, MaskTile selTmpTile)
{
    constexpr float scalarHalf = 0.5f;
    constexpr float scalarNegTwo = -2.0f;
    constexpr float scalarNegOne = -1.0f;
    constexpr float scalarOne = 1.0f;
    constexpr float scalarZero = 0.0f;
    const float nanValue = __builtin_nanf("");
    const float infValue = __builtin_huge_valf();
    const float negInfValue = -infValue;

    // mag = exp(y * log(|x|)) -> dst
    pto::TABS(tmp0Tile, src0Tile);
    SyncV();
    pto::TLOG(tmp0Tile, tmp0Tile);
    SyncV();
    pto::TMUL(tmp0Tile, src1Tile, tmp0Tile);
    SyncV();
    pto::TEXP(dstTile, tmp0Tile);
    SyncV();

    // parity = y - 2 * floor(0.5 * y) -> tmp0; odd integer exponent = (parity == 1) -> mask0
    pto::TMULS(tmp1Tile, src1Tile, scalarHalf);
    SyncV();
    pto::TCVT(tmp1Tile, tmp1Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TMULS(tmp1Tile, tmp1Tile, scalarNegTwo);
    SyncV();
    pto::TADD(tmp0Tile, src1Tile, tmp1Tile);
    SyncV();
    pto::TCMPS(mask0Tile, tmp0Tile, scalarOne, pto::CmpMode::EQ);
    SyncV();

    // negative base -> mask1
    pto::TCMPS(mask1Tile, src0Tile, scalarZero, pto::CmpMode::LT);
    SyncV();

    // x < 0 and odd integer exponent: dst = -mag (mask0 = xneg AND isOdd, in-place)
    pto::TAND(mask0Tile, mask1Tile, mask0Tile);
    SyncV();
    pto::TMULS(tmp0Tile, dstTile, scalarNegOne);
    SyncV();
    pto::TSEL(dstTile, mask0Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // floor(y) -> tmp1; non-integer exponent = (y != floor(y)) -> mask0 (reused)
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCMP(mask0Tile, src1Tile, tmp1Tile, pto::CmpMode::NE);
    SyncV();

    // x < 0 (finite) and non-integer exponent: dst = NaN.
    // -inf ^ non-integer = +mag (already in dst), so exclude x == -inf via x > -inf.
    pto::TAND(mask1Tile, mask1Tile, mask0Tile); // mask1 = xneg AND !isInt
    SyncV();
    pto::TCMPS(mask0Tile, src0Tile, negInfValue, pto::CmpMode::GT);
    SyncV();
    pto::TAND(mask1Tile, mask1Tile, mask0Tile); // mask1 = xneg AND !isInt AND x > -inf
    SyncV();
    pto::TEXPANDS(tmp0Tile, nanValue);
    SyncV();
    pto::TSEL(dstTile, mask1Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // y == 0: dst = 1
    pto::TCMPS(mask0Tile, src1Tile, scalarZero, pto::CmpMode::EQ);
    SyncV();
    pto::TEXPANDS(tmp0Tile, scalarOne);
    SyncV();
    pto::TSEL(dstTile, mask0Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // x == 1: dst = 1 (tmp0 still holds 1)
    pto::TCMPS(mask0Tile, src0Tile, scalarOne, pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(dstTile, mask0Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // x == -1 and y is +/-inf: dst = 1 (mag is NaN because log(1) == 0)
    pto::TABS(tmp1Tile, src1Tile);
    SyncV();
    pto::TCMPS(mask0Tile, src0Tile, scalarNegOne, pto::CmpMode::EQ);
    SyncV();
    pto::TCMPS(mask1Tile, tmp1Tile, infValue, pto::CmpMode::EQ);
    SyncV();
    pto::TAND(mask1Tile, mask0Tile, mask1Tile); // mask1 = x==-1 AND |y|==inf
    SyncV();
    pto::TSEL(dstTile, mask1Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();
}
#endif

#define OP_TILE_OP_POW TPow
template <auto PrecisionType = pto::PowAlgorithm::DEFAULT, typename T0, typename T1, typename T2, typename T3>
TILEOP void TPow(T0 dst, T1 src0, T2 src1, T3 tmp)
{
#ifdef __DAV_V220
    if constexpr (std::is_same_v<typename T0::Type, float>) {
        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
        auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
        if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0) {
            return;
        }
        constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
        constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
        constexpr auto dataTypeSize = sizeof(typename T0::Type);
        constexpr auto floatSlot = tileH * tileW * dataTypeSize;    // bytes, 32B-aligned
        constexpr auto maskCols = ((tileW + 7) / 8 + 31) / 32 * 32; // bit-packed mask bytes/row, 32B-aligned
        constexpr auto maskSlot = tileH * maskCols;                 // bytes
        constexpr auto maskBase = 2 * floatSlot;                    // masks laid out after 2 float tiles
        using DataTile = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, maskCols, pto::BLayout::RowMajor, -1, -1>;
        DataTile dstTile(dstShape3, dstShape4);
        DataTile src0Tile(dstShape3, dstShape4);
        DataTile src1Tile(dstShape3, dstShape4);
        DataTile tmp0Tile(dstShape3, dstShape4);
        DataTile tmp1Tile(dstShape3, dstShape4);
        MaskTile mask0Tile(dstShape3, (dstShape4 + 7) / 8);
        MaskTile mask1Tile(dstShape3, (dstShape4 + 7) / 8);
        MaskTile selTmpTile(dstShape3, (dstShape4 + 7) / 8);
        for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    auto dstOffset = GenTileOffset(dst, tileOffsets);
                    auto src0Offset = GenTileOffset(src0, tileOffsets);
                    auto src1Offset = GenTileOffset(src1, tileOffsets);
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dataTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * dataTypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * dataTypeSize));
                    pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + 0 * floatSlot));
                    pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + 1 * floatSlot));
                    pto::TASSIGN(mask0Tile, (uint64_t)(tmp.GetAddr() + maskBase + 0 * maskSlot));
                    pto::TASSIGN(mask1Tile, (uint64_t)(tmp.GetAddr() + maskBase + 1 * maskSlot));
                    pto::TASSIGN(selTmpTile, (uint64_t)(tmp.GetAddr() + maskBase + 2 * maskSlot));
                    TPowFloatTile(dstTile, src0Tile, src1Tile, tmp0Tile, tmp1Tile, mask0Tile, mask1Tile, selTmpTile);
                }
            }
        }
        return;
    }
#endif
    BinaryTmpCompute<BinaryOp::POW, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REM TRem
template <auto PrecisionType = pto::RemAlgorithm::DEFAULT, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRemainder(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BinaryTmpCompute<BinaryOp::REM, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_AXPY TAxpy
template <int... BrcOperands, typename T0, typename T1, typename Scalar>
TILEOP void TAxpy(T0 dst, T1 src0, Scalar alpha)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    if constexpr (!TileOp::HasBrcOperand<BrcOperands...>()) {
        auto dstTile = PtoTile<T0>(dst);
        auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    pto::TAXPY(dstTile.Data(), src0ExecTile, static_cast<typename T1::Type>(alpha));
                }
            }
        }
        return;
    }

    using SrcTileInfo = TensorTileInfo<T1>;

    using SrcPtoTile = typename std::conditional<(SrcTileInfo::tileW == 1 &&
                                                  GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT),
                                                 PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = SrcPtoTile(src0);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dsttileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto src0tileOffsets = TileOffset(
                    (SrcTileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : n0Index,
                    (SrcTileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : n1Index,
                    (SrcTileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 : n2Index);
                dstTile.Assign(dst, dsttileOffsets);
                src0Tile.Assign(src0, src0tileOffsets);
                pto::TAXPY(dstTile.Data(), src0Tile.Data(), static_cast<typename T1::Type>(alpha));
            }
        }
    }
}

template <auto pos, auto neg, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7>
TILEOP void IntFloorDiv(T0 dst, T1 src0, T2 src1, T3 tmp0, T4 tmp1, T5 tmp2, T6 tmp3, T7 tmp4)
{
    // MaskTile: tmp3, tmp4
    // DataTile: tmp0-tmp2
    // reuse tmp address: tmp2=tmp4

    // Deal dividend is zero
    pto::TCMPS(tmp3, src0, 0, pto::CmpMode::LT);
    pto::TSELS(tmp2, tmp3, tmp2, tmp2, pos);
    pto::TCMPS(tmp3, src0, 0, pto::CmpMode::GE);
    pto::TSELS(tmp2, tmp3, tmp2, tmp2, neg);
    pto::TCMPS(tmp3, src1, 0, pto::CmpMode::NE);
    pto::TMOV(tmp0, src0);
    pto::TSEL(tmp0, tmp3, tmp0, tmp2, tmp2);
    pto::TSELS(tmp1, tmp3, src1, tmp2, 1);

    /*
     * After zero-divisor handling:
     * sign_differ = (src0 < 0) != (src1 < 0)
     * quot = src0 / src1
     * rem = src0 - quot * src1
     * dst = (sign_differ && rem != 0) ? quot - 1 : quot
     */
    pto::TCMPS(tmp3, tmp0, 0, pto::CmpMode::LT);
    pto::TCMPS(tmp4, tmp1, 0, pto::CmpMode::LT);
    pto::TXOR(tmp3, tmp3, tmp4, tmp4);

    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(dst, tmp0, tmp1);

    // A5 TREM does not use the tmp argument.
    pto::TREM(tmp0, tmp0, tmp1, tmp0);

    pto::TCMPS(tmp4, tmp0, 0, pto::CmpMode::NE);
    pto::TAND(tmp3, tmp3, tmp4);
    pto::TADDS(tmp0, dst, -1);
    pto::TSEL(dst, tmp3, tmp0, dst, tmp2);
}

template <typename TmpTensor>
TILEOP uint64_t FloorDivTmpAddr(TmpTensor tmp, size_t byteOffset, size_t tileShapeSize, size_t tileIndex,
                                size_t elementSize)
{
    return (uint64_t)(tmp.GetAddr() + byteOffset + tileIndex * tileShapeSize * elementSize);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivFp32TmpCompute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                   size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, tmp1Tile);
    SyncV();
    pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_RINT);
    SyncV();
}

template <typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivFloatCompute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile)
{
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, src1Tile);
    SyncV();
    pto::TCVT(dstTile, dstTile, pto::RoundMode::CAST_FLOOR);
    SyncV();
}

#ifdef __DAV_V220
template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivV220Int8Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                    size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp3Tile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();
    pto::TCVT(tmp0Tile, tmp2Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::ON);
    SyncV();
}
template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivV220Int32Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                     size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;

    Fp32TileDefine tmp0Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Fp32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp0I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp2I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp3I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp4I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp5I32Tile(dstShape3, dstShape4);
    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);

    pto::TASSIGN(tmp0Fp32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1Fp32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2Fp32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp0I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp2I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp3I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp4I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(int32_t)));
    pto::TASSIGN(tmp5I32Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 5, sizeof(int32_t)));
    pto::TASSIGN(tmp1MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));

    // Step 1: approximate quotient by float32 division, then floor and cast to int32.
    // q = floor(float32(x1) / float32(x2))
    pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCVT(tmp1Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Fp32Tile, tmp0Fp32Tile, tmp1Fp32Tile);
    SyncV();
    pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // Step 2: compute exact int32 remainder: r = x1 - q * x2.
    pto::TMUL(tmp0I32Tile, dstTile, src1Tile);
    SyncV();
    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
    SyncV();

    // Step 3: refine q with floor(float32(r) / float32(x2)).
    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Fp32Tile, tmp2Fp32Tile, tmp1Fp32Tile);
    SyncV();
    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // Step 4: apply the remainder-based correction.
    // q_corrected = q + correction
    pto::TADD(dstTile, dstTile, tmp0I32Tile);
    SyncV();

    // Step 5: recompute r2 with q_corrected.
    // A valid floor-div remainder must satisfy 0 <= r2 * sign(x2) < abs(x2).
    pto::TMUL(tmp0I32Tile, dstTile, src1Tile);
    SyncV();
    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
    SyncV();

    // Step 6: final +/-1 correction.
    // Use float32 only to produce comparison masks; keep abs(x2) and diff in int32.
    pto::TCVT(tmp2Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();

    // Normalize remainder: r2_adj = (x2 < 0) ? -r2 : r2.
    pto::TMULS(tmp2I32Tile, tmp0I32Tile, -1);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp3I32Tile);
    SyncV();

    // Compute abs(x2) in int32 using the same sign mask.
    pto::TMULS(tmp2I32Tile, src1Tile, -1);
    SyncV();
    pto::TSEL(tmp2I32Tile, tmp1MaskTile, tmp2I32Tile, src1Tile, tmp3I32Tile);
    SyncV();

    pto::TSUB(tmp3I32Tile, tmp0I32Tile, tmp2I32Tile); // diff = r2_adj - abs(x2)
    SyncV();

    // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
    // tensor-scalar select path, whose first lane can be unstable across calls.
    pto::TSUB(tmp4I32Tile, tmp2I32Tile, tmp2I32Tile); // zero
    SyncV();

    // If r2_adj < 0, q_corrected is too large: final_corr = -1; otherwise 0.
    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();
    pto::TADDS(tmp2I32Tile, tmp4I32Tile, -1);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp4I32Tile, tmp5I32Tile);
    SyncV();

    // If diff >= 0, q_corrected is too small: final_corr = 1.
    pto::TCVT(tmp2Fp32Tile, tmp3I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
    SyncV();
    pto::TADDS(tmp2I32Tile, tmp4I32Tile, 1);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp5I32Tile);
    SyncV();

    // res = q_corrected + final_corr
    pto::TADD(dstTile, dstTile, tmp0I32Tile);
    SyncV();
}
#else
template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivNonV220Uint8Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                        size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Int16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    Int16TileDefine tmp2Tile(dstShape3, dstShape4);
    Int16TileDefine tmp3Tile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);
    pto::TCVT(dstTile, tmp2Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::ON);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivNonV220Int8Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                       size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Uint8TileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefine tmp0DataTile(dstShape3, dstShape4);
    HalfTileDefine tmp1DataTile(dstShape3, dstShape4);
    Uint8TileDefine tmp2MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));

    pto::TCVT(tmp0DataTile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1DataTile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCMPS(tmp2MaskTile, tmp1DataTile, 0, pto::CmpMode::NE);
    pto::TSELS(tmp0DataTile, tmp2MaskTile, tmp0DataTile, tmp0DataTile, 0);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0DataTile, tmp0DataTile, tmp1DataTile);
    pto::TCVT(dstTile, tmp0DataTile, pto::RoundMode::CAST_FLOOR);
}

template <typename T0, typename T3, auto tileH, auto tileW, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void FloorDivNonV220Int32Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T3 tmp, size_t offset,
                                        size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Uint8TileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;
    Int32TileDefine tmp0DataTile(dstShape3, dstShape4);
    Int32TileDefine tmp1DataTile(dstShape3, dstShape4);
    Int32TileDefine tmp2DataTile(dstShape3, dstShape4);
    Uint8TileDefine tmp3MaskTile(dstShape3, dstShape4);
    Uint8TileDefine tmp4MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2DataTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp4MaskTile, FloorDivTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));

    IntFloorDiv<(int32_t)0x7FFF7F7F, (int32_t)0x80008080>(dstTile, src0Tile, src1Tile, tmp0DataTile, tmp1DataTile,
                                                          tmp2DataTile, tmp3MaskTile, tmp4MaskTile);
}
#endif

#define OP_TILE_OP_FLOORDIV TFloorDiv
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TFloorDiv(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    DataTileDefine dstTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T3::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                if constexpr (std::is_same_v<typename T0::Type, half> ||
                              std::is_same_v<typename T0::Type, bfloat16_t>) {
                    FloorDivFp32TmpCompute<T0, T3, tileH, tileW>(dstTile, src0ExecTile, src1ExecTile, tmp,
                                                                 tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, float>) {
                    FloorDivFloatCompute(dstTile, src0ExecTile, src1ExecTile);
                }

#ifdef __DAV_V220
                if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    FloorDivV220Int32Compute<T0, T3, tileH, tileW>(dstTile, src0ExecTile, src1ExecTile, tmp,
                                                                   tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t> ||
                                     std::is_same_v<typename T0::Type, uint8_t>) {
                    FloorDivV220Int8Compute<T0, T3, tileH, tileW>(dstTile, src0ExecTile, src1ExecTile, tmp,
                                                                  tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                }
#else
                if constexpr (std::is_same_v<typename T0::Type, uint8_t>) {
                    FloorDivNonV220Uint8Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t>) {
                    FloorDivNonV220Int8Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    FloorDivNonV220Int32Compute<T0, T3, tileH, tileW>(
                        dstTile, src0ExecTile, src1ExecTile, tmp, tmpByteOffset, dstShape3, dstShape4, tileShapeSize);
                }
#endif
            }
        }
    }
}
#endif
