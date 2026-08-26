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
 * \file binary_scalar.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#define TILEOP_TILE_OPERATOR_BINARY_SCALAR__H
#include "binary.h"

template <BinaryScalarOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == BinaryScalarOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::SUB) {
        if constexpr (std::is_same<Scalar, half>::value) {
            PTO_WITH_LAST_USE(
                pto::TADDS(dst, src0, static_cast<half>(static_cast<float>(-1) * static_cast<float>(src1))), n1, n2);
        } else {
            PTO_WITH_LAST_USE(pto::TADDS(dst, src0, -src1), n1, n2);
        }
        return;
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS<PrecisionType>(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src0, src1), n1, n2);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEAND) {
        pto::TANDS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::BITWISEOR) {
        pto::TORS(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::MOD) {
        pto::TFMODS<PrecisionType>(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryScalarOp::LRELU) {
        pto::TLRELU(dst, src0, src1);
        return;
    }
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }

    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                BinaryScalarComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), src0ExecTile, src1);
            }
        }
    }
}

#define OP_TILE_OP_ADDS TAddS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::ADD, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::SUB, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MUL, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIVS TDivS
template <auto PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename Scalar,
          typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::DIV, PrecisionType, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAXS TMaxS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMaxS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MAX, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MINS TMinS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMinS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MIN, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_LRELU TLReLU
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TLReLU(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::LRELU, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEANDS TBitwiseAndS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseAndS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::BITWISEAND, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEORS TBitwiseOrS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseOrS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::BITWISEOR, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_GCDS TGcdS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TGcdS(T0 dst, T1 src0, Scalar src1, T2 temp)
{
    // 统一 int32 计算：q*b 可达 a+b（int16 输入时 65534 超 int16），且仿真器上 int16 指令不可靠
    using CalcType = int32_t;
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
    const auto srcLayout = src0.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }

    // temp 布局（与 BinaryOperationScalarTileFunc 中 GCDS 的临时空间分配保持一致）：
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
                auto src0Offset = GenTileOffset(src0, tileOffsets);
                TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(typename T0::Type)));
                TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * sizeof(typename T1::Type)));

                // 统一转换到计算位宽。a2/a3 的 TCVT 没有 int8/uint8/int16 到 int32 的
                // 直接转换，int16 经 f32、int8/uint8 经 f16 中转（小整数转换无精度损失）
                // b = |scalar| 广播到整 tile；a 按输入类型转换到计算位宽
                pto::TEXPANDS(bTile, static_cast<CalcType>(src1));
                if constexpr (Std::is_same_v<typename T0::Type, int32_t>) {
                    pto::TMOV(aTile, src0Tile);
                } else if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    pto::TCVT(fTile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(aTile, fTile, pto::RoundMode::CAST_RINT);
                } else {
                    // int8/uint8: s8/u8 -> f16 -> s16（f16 中转视图复用 fbTile 的地址）
                    F16Tile f16Tile(shape3, shape4);
                    TASSIGN(f16Tile, temp.GetAddr() + fbBase);
                    pto::TCVT(f16Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(aTile, f16Tile, pto::RoundMode::CAST_RINT);
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

#define OP_TILE_OP_MODS TModS
template <auto PrecisionType = pto::FmodSAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename Scalar,
          typename T0, typename T1>
TILEOP void TModS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MOD, PrecisionType, LastUse>(dst, src0, src1);
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpComputeImpl(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    if constexpr (op == BinaryScalarOp::BITWISEXOR) {
        pto::TXORS(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryScalarOp::REM) {
        pto::TREMS<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryScalarOp::POW) {
        pto::TPOWS<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename T0, typename T1, typename Scalar, typename T2>
TILEOP void BinaryScalarTmpCompute(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    if constexpr (op == BinaryScalarOp::REM) {
        auto tmpTile = PtoTile<T2>(tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    tmpTile.Assign(tmp, tileOffsets);
                    BinaryScalarTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1, tmpTile.Data());
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
                    AssignElementwiseOperandExecTile(tmpExecTile, tmp, tileOffsets);
                    BinaryScalarTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1, tmpExecTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXORS TBitwiseXorS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TBitwiseXorS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    BinaryScalarTmpCompute<BinaryScalarOp::BITWISEXOR, 0>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REMS TRemainderS
template <typename Scalar, auto PrecisionType = pto::RemSAlgorithm::DEFAULT, typename T0, typename T1, typename T2>
TILEOP void TRemainderS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    BinaryScalarTmpCompute<BinaryScalarOp::REM, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_POWS TPowS
template <auto PrecisionType = pto::PowAlgorithm::DEFAULT, typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TPowS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    BinaryScalarTmpCompute<BinaryScalarOp::POW, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REMRS TRemainderRS
template <typename Scalar, auto PrecisionType = pto::RemAlgorithm::DEFAULT, typename T0, typename T1, typename T2>
TILEOP void TRemainderRS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    using tmp0TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                     -1, -1>;
    using tmp1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 2, tmpTileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    tmp0TileDefine tmp0Tile(shape3, shape4);
    tmp1TileDefine tmp1Tile(2, shape4);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + shape3 * tmpTileW * sizeof(typename T2::Type)));
                pto::TEXPANDS(tmp0Tile, src1);
#ifdef __DAV_V220
                pipe_barrier(PIPE_V);
#endif
                pto::TREM<PrecisionType>(dstTile.Data(), tmp0Tile, src0ExecTile, tmp1Tile);
            }
        }
    }
}

#define OP_TILE_OP_FLOORDIVS TFloorDivS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TFloorDivS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto tileShapeSize = TileOp::GetAnyAxisMergeResult<
        DIM_1ST, Std::tuple_size<typename T0::TileShape>::value, typename T0::TileShape>();

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    DataTileDefine src0Tile(dstShape3, dstShape4);
    DataTileDefine dstTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto srcOffset = GenTileOffset(src0, tileOffsets);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + srcOffset * dstTypeSize));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                if constexpr (std::is_same_v<typename T0::Type, half> ||
                              std::is_same_v<typename T0::Type, bfloat16_t>) {
                    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, static_cast<float>(src1));
                    SyncV();
                    pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_RINT);
                    SyncV();
                } else if constexpr (std::is_same_v<typename T0::Type, float>) {
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, static_cast<float>(src1));
                    SyncV();
                    pto::TCVT(dstTile, dstTile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                }

#ifdef __DAV_V220
                if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor,
                                                      -1, -1>;
                    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW,
                                                     pto::BLayout::RowMajor, -1, -1>;

                    Fp32TileDefine tmp0Fp32Tile(dstShape3, dstShape4);
                    Fp32TileDefine tmp2Fp32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp0I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp2I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp3I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp4I32Tile(dstShape3, dstShape4);
                    Int32TileDefine tmp5I32Tile(dstShape3, dstShape4);
                    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);

                    pto::TASSIGN(tmp0Fp32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int32_t)));
                    pto::TASSIGN(tmp2Fp32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int32_t)));
                    pto::TASSIGN(tmp0I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int32_t)));
                    pto::TASSIGN(tmp2I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int32_t)));
                    pto::TASSIGN(tmp3I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 3, sizeof(int32_t)));
                    pto::TASSIGN(tmp4I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 4, sizeof(int32_t)));
                    pto::TASSIGN(tmp5I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 5, sizeof(int32_t)));
                    pto::TASSIGN(tmp1MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(int32_t)));
                    auto divisor = static_cast<int32_t>(src1);

                    // Step 1: approximate quotient by float32 division, then floor and cast to int32.
                    // q = floor(float32(x1) / float32(x2))
                    pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Fp32Tile, tmp0Fp32Tile,
                                                                  static_cast<float>(divisor));
                    SyncV();
                    pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();

                    // Step 2: compute exact int32 remainder: r = x1 - q * x2.
                    pto::TMULS(tmp0I32Tile, dstTile, divisor);
                    SyncV();
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
                    SyncV();

                    // Step 3: refine q with floor(float32(r) / float32(x2)).
                    pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Fp32Tile, tmp2Fp32Tile,
                                                                  static_cast<float>(divisor));
                    SyncV();
                    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();

                    // Step 4: apply the remainder-based correction.
                    // q_corrected = q + correction
                    pto::TADD(dstTile, dstTile, tmp0I32Tile);
                    SyncV();

                    // Step 5: recompute r2 with q_corrected.
                    pto::TMULS(tmp0I32Tile, dstTile, divisor);
                    SyncV();
                    pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
                    SyncV();

                    // Step 6: final +/-1 correction. A valid floor-div remainder must satisfy
                    // 0 <= r2 * sign(x2) < abs(x2).
                    auto absSrc1 = divisor;
                    if (divisor < 0) {
                        pto::TMULS(tmp0I32Tile, tmp0I32Tile, -1); // r2_adj = -r2
                        SyncV();
                        absSrc1 = -divisor;
                    }

                    pto::TADDS(tmp3I32Tile, tmp0I32Tile, -absSrc1); // diff = r2_adj - abs(x2)
                    SyncV();

                    // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
                    // tensor-scalar select path, whose first lane can be unstable across calls.
                    pto::TSUB(tmp4I32Tile, tmp0I32Tile, tmp0I32Tile); // zero
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
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t> ||
                                     std::is_same_v<typename T0::Type, uint8_t>) {
                    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1,
                                                     -1>;
                    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor,
                                                     -1, -1>;
                    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
                    Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(tmp1Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp1Tile, tmp1Tile, static_cast<float>(src1));
                    SyncV();
                    pto::TCVT(tmp0Tile, tmp1Tile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::ON);
                    SyncV();
                }
#else
                if constexpr (std::is_same_v<typename T0::Type, uint8_t>) {
                    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1,
                                                     -1>;
                    using Int16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor,
                                                      -1, -1>;
                    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
                    Int16TileDefine tmp1Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(tmp1Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
                    pto::TDIVS(tmp1Tile, tmp1Tile, static_cast<int16_t>(src1));
                    pto::TCVT(dstTile, tmp1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::ON);
                } else if constexpr (std::is_same_v<typename T0::Type, int8_t>) {
                    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1,
                                                     -1>;
                    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    if (src1 == 0) {
                        pto::TEXPANDS(tmp0Tile, static_cast<half>(0.0f));
                    } else {
                        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile,
                                                                      static_cast<half>(static_cast<float>(src1)));
                    }
                    pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
                } else if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor,
                                                      -1, -1>;
                    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW,
                                                     pto::BLayout::RowMajor, -1, -1>;
                    Int32TileDefine tmp0DataTile(dstShape3, dstShape4);
                    Int32TileDefine tmp1DataTile(dstShape3, dstShape4);
                    MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
                    MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
                    pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
                    pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
                    pto::TASSIGN(tmp2MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(float)));
                    pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));

                    if (src1 == 0) {
                        constexpr int32_t pos = 0x7FFF7F7F;
                        constexpr int32_t neg = 0x80008080;
                        pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
                        pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, pos);
                        pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
                        pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, neg);
                    } else {
                        if (src1 < 0) {
                            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
                        } else {
                            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
                        }
                        pto::TDIVS(dstTile, src0Tile, static_cast<int32_t>(src1));
                        pto::TMULS(tmp0DataTile, dstTile, static_cast<int32_t>(src1));
                        pto::TSUB(tmp0DataTile, src0Tile, tmp0DataTile);
                        pto::TCMPS(tmp3MaskTile, tmp0DataTile, 0, pto::CmpMode::NE);
                        pto::TAND(tmp2MaskTile, tmp2MaskTile, tmp3MaskTile);
                        pto::TADDS(tmp1DataTile, dstTile, -1);
                        pto::TSEL(dstTile, tmp2MaskTile, tmp1DataTile, dstTile, tmp0DataTile);
                    }
                }
#endif
            }
        }
    }
}
#endif
