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
 * \file sincos.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_SINCOS__H
#define TILEOP_TILE_OPERATOR_SINCOS__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cmath>

const uint8_t HALF_CALC_PROCEDURE = 4;
const uint8_t FLOAT_NOREUSE_CALC_PROCEDURE = 3;
const uint8_t FLOAT_REUSE_CALC_PROCEDURE = 2;

// define the number of x div pi
constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;
// define the PI for compute
constexpr float PI_V2 = 3.140625;
constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375;
constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
constexpr float KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13;
// define the number of down of pi_div
constexpr float PI_DOWN = 1.57079637050628662109375;
// kpi_2
constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;
constexpr float RES_MULTI_SCA = 2.604926501e-6;
constexpr float RES_ADDICT_UP = -0.0001980894471;
constexpr float ADD2S = 0.008333049340;
constexpr float ADD3S = -0.1666665792;
constexpr float POINT_FIVE = 0.5;
constexpr float M4_SCA = 4.0;
constexpr float K2_SCA = -2.0;
constexpr float TRIG_ONE = 1.0;
constexpr float TRIG_ZERO = 0.0;

template <TrigtOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void reduceKCompute(T0 dst, T1 src0, T2 tmp0, T3 tmp1)
{
    pto::TMULS(tmp0, src0, TRIG_ZERO);
    SyncV();
    pto::TADD(src0, src0, tmp0);
    SyncV();
    //  k=round(x/π), x0=x-kπ, x0 belongs to [-π/2, π/2]
    //  cos(x) = (-1)^k * sin(x0 + π/2)
    pto::TMULS(tmp0, src0, PI_FOR_X_TODIV);
    SyncV();
    if constexpr (op == TrigtOp::SIN) {
        pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_ROUND);
        SyncV();
    }
    if constexpr (op == TrigtOp::COS) {
        pto::TADDS(tmp0, tmp0, POINT_FIVE);
        SyncV();
        pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_RINT);
        SyncV();
    }
    pto::TCVT(tmp0, tmp1, pto::RoundMode::CAST_NONE);
    SyncV();

    // x -= k * pi_0
    pto::TMULS(dst, tmp0, PI_V2);
    SyncV();
    pto::TSUB(src0, src0, dst);
    SyncV();
    // x -= k * pi_1
    pto::TMULS(dst, tmp0, KPI_FIRS_PI_MULS);
    SyncV();
    pto::TSUB(src0, src0, dst);
    SyncV();
    // x = x + PI_DOWN
    if constexpr (op == TrigtOp::COS) {
        pto::TADDS(src0, src0, PI_DOWN);
        SyncV();
    }
    // x -= k * pi_2
    pto::TMULS(dst, tmp0, KPI_TWI_PI_MULS);
    SyncV();
    pto::TSUB(src0, src0, dst);
    SyncV();
    // x -= k * pi_3
    pto::TMULS(dst, tmp0, KPI_THIR_PI_MULS);
    SyncV();
    pto::TSUB(src0, src0, dst);
    SyncV();

    // x -= k * pi_4
    pto::TMULS(dst, tmp0, KPI_FOR_PI_MULS);
    SyncV();
    pto::TSUB(src0, src0, dst);
    SyncV();

    if constexpr (op == TrigtOp::COS) {
        // x = x + PI_RESDOWN_ADDS_NEG
        pto::TADDS(src0, src0, PI_RESDOWN_ADDS_NEG);
        SyncV();
    }
    // kover2
    pto::TMULS(dst, tmp0, POINT_FIVE);
    SyncV();

    pto::TCVT(tmp1, dst, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dst, tmp1, pto::RoundMode::CAST_NONE);
    SyncV();
    // kover2floorm4
    pto::TMULS(dst, dst, M4_SCA);
    SyncV();
    //k2
    pto::TMULS(tmp0, tmp0, K2_SCA);
    SyncV();
    //sign
    pto::TADD(dst, dst, tmp0);
    SyncV();
    pto::TADDS(dst, dst, TRIG_ONE);
    SyncV();
}

template <TrigtOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void SinCosCompute(T0 dst, T1 src0, T2 tmp0, T3 tmp1)
{
    // x^2
    pto::TMUL(tmp0, src0, src0);
    SyncV();
    // sin(x) = x * P(x)
    // P(x) = (((x^2 * R0 + R1) * x^2 + R2) * x^2 + R3) * x^2 + 1.0
    // roundTensor = mul(x^2, 2.604926501e-6)
    pto::TMULS(tmp1, tmp0, RES_MULTI_SCA);
    SyncV();
    pto::TADDS(tmp1, tmp1, RES_ADDICT_UP);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, ADD2S);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, ADD3S);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, TRIG_ONE);
    SyncV();
    // sin(x) = x * P(x)
    pto::TMUL(tmp1, src0, tmp1);
    SyncV();
    pto::TMUL(dst, dst, tmp1);
    SyncV();
    return;
}

template <TrigtOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void TrigCompute(T0 dst, T1 tmp0, T2 tmp1, T3 src0)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto tmp1TileH = TileOp::GetTensorTileShapeDim<T3, 3, 5>();
    constexpr auto tmp1TileW = TileOp::GetTensorTileShapeDim<T3, 4, 5>();
    auto dstTile = PtoTile<T0>(dst);
    auto tmp0Tile = PtoTile<T1>(tmp0);
    auto tmp1Tile = PtoTile<T2>(tmp1);
    auto src0Tile = PtoTile<T3>(src0);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                tmp0Tile.Assign(tmp0, tileOffsets);
                tmp1Tile.Assign(tmp1, tileOffsets);
                reduceKCompute<op>(dstTile.Data(), src0Tile.Data(), tmp0Tile.Data(), tmp1Tile.Data());
                SyncV();
                using TmpTile = pto::Tile<pto::TileType::Vec, typename T3::Type, tmp1TileH, tmp1TileW, pto::BLayout::RowMajor, -1, -1>;
                TmpTile tmp2Tile(shape3, shape4);
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp1.GetAddr()));
                SinCosCompute<op>(dstTile.Data(), src0Tile.Data(), tmp0Tile.Data(), tmp2Tile);
            }
        }
    }
}

#define OP_TILE_OP_SIN TSin
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TSin(T0 dst, T1 tmp0, T2 tmp1, T3 src0)
{
    TrigCompute<TrigtOp::SIN>(dst, tmp0, tmp1, src0);
}

#define OP_TILE_OP_COS TCos
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TCos(T0 dst, T1 tmp0, T2 tmp1, T3 src0)
{
    TrigCompute<TrigtOp::COS>(dst, tmp0, tmp1, src0);
}
#endif