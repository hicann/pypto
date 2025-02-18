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
 * \file sin.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor Sin(Tensor operand) {
    // An algorithm guarante data precision from -10^10 to 10^10
    auto dType = operand.GetStorage()->Datatype();
    if (dType != DataType::DT_FP32) {
        operand = Cast(operand, DataType::DT_FP32);
    }

    constexpr float number2048 = 2048.0f;
    constexpr float oneOverN = 1.0 / 2048.0;
    constexpr float invHalfPi = 0.63661975f;
    constexpr float pi0 = 1.5708008f;
    constexpr float pi1 = -0.0000044535846f;
    constexpr float pi2 = -8.706138e-10f;
    constexpr float F_025 = 0.25;
    constexpr float F_05 = 0.5;
    constexpr float F_4 = 4.0;
    constexpr float F_1 = 1.0;
    constexpr float F_NEGA_1 = -1.0;
    constexpr float F_NEGA_2 = -2.0;

    auto xScaled = Mul(operand, Element(DataType::DT_FP32, oneOverN));
    auto xOverpi = Mul(xScaled, Element(DataType::DT_FP32, invHalfPi));
    auto n = Cast(xOverpi, DataType::DT_FP32, CAST_ROUND);
    auto n0 = Mul(xOverpi, Element(DataType::DT_FP32, oneOverN));
    n0 = Cast(n0, DataType::DT_FP32, CAST_ROUND);
    n0 = Mul(n0, Element(DataType::DT_FP32, number2048));

    auto n1 = Sub(n, n0);

    auto fix = Mul(n0, Element(DataType::DT_FP32, pi0));
    auto xFix = Sub(xScaled, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, pi0));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, pi1));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, pi1));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, pi2));
    xFix = Sub(xFix, fix);

    constexpr float PI_02 = 1.5703125f;
    constexpr float PI_12 = 0.0004837513f;

    auto remainX = Mul(xFix, Element(DataType::DT_FP32, number2048));
    auto temp = Mul(remainX, Element(DataType::DT_FP32, invHalfPi));
    auto n2 = Cast(temp, DataType::DT_FP32, CAST_ROUND);

    n0 = Mul(n0, Element(DataType::DT_FP32, number2048));
    n1 = Mul(n1, Element(DataType::DT_FP32, number2048));
    fix = Mul(n0, Element(DataType::DT_FP32, PI_02));
    xFix = Sub(operand, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_02));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, PI_12));
    xFix = Sub(xFix, fix);

    constexpr float PI_22 = 0.000000075495336f;
    fix = Mul(n2, Element(DataType::DT_FP32, PI_02));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_12));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, PI_22));
    xFix = Sub(xFix, fix);

    constexpr float PI_32 = 2.5579538e-12f;
    fix = Mul(n2, Element(DataType::DT_FP32, PI_12));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_22));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, PI_32));
    xFix = Sub(xFix, fix);

    constexpr float PI_42 = 5.389786e-15f;
    fix = Mul(n2, Element(DataType::DT_FP32, PI_22));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_32));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, PI_42));
    xFix = Sub(xFix, fix);

    constexpr float PI_52 = 5.166901e-19f;
    fix = Mul(n2, Element(DataType::DT_FP32, PI_32));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_42));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, PI_52));
    xFix = Sub(xFix, fix);

    constexpr float PI_62 = 3.281839e-22f;
    fix = Mul(n2, Element(DataType::DT_FP32, PI_42));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_52));
    xFix = Sub(xFix, fix);
    fix = Mul(n0, Element(DataType::DT_FP32, PI_62));
    xFix = Sub(xFix, fix);

    fix = Mul(n2, Element(DataType::DT_FP32, PI_52));
    xFix = Sub(xFix, fix);
    fix = Mul(n1, Element(DataType::DT_FP32, PI_62));
    xFix = Sub(xFix, fix);
    fix = Mul(n2, Element(DataType::DT_FP32, PI_62));
    xFix = Sub(xFix, fix);

    auto halfN2 = Mul(n2, Element(DataType::DT_FP32, F_05));
    auto half4N2 = Mul(n2, Element(DataType::DT_FP32, F_025));
    auto nHalf2 = Cast(halfN2, DataType::DT_FP32, CAST_FLOOR);
    auto nHalf4 = Cast(half4N2, DataType::DT_FP32, CAST_FLOOR);

    auto k1 = Mul(nHalf2, Element(DataType::DT_FP32, F_NEGA_2));
    auto k2 = Mul(nHalf4, Element(DataType::DT_FP32, F_4));
    auto sign = Add(k1, k2);
    sign = Add(sign, Element(DataType::DT_FP32, F_1));

    auto ifcos = Add(n2, k1);
    auto ifsin = Mul(ifcos, Element(DataType::DT_FP32, F_NEGA_1));
    ifsin = Add(ifsin, Element(DataType::DT_FP32, F_1));

    constexpr float scoef4 = 0.0000027183114939898219064f;
    constexpr float scoef3 = -0.000198393348360966317347f;
    constexpr float scoef2 = 0.0083333293858894631756f;
    constexpr float scoef1 = -0.166666666416265235595f;
    auto xPow = Mul(xFix, xFix);
    auto sinPoly = Mul(xPow, Element(DataType::DT_FP32, scoef4));
    sinPoly = Add(sinPoly, Element(DataType::DT_FP32, scoef3));
    sinPoly = Mul(xPow, sinPoly);
    sinPoly = Add(sinPoly, Element(DataType::DT_FP32, scoef2));
    sinPoly = Mul(xPow, sinPoly);
    sinPoly = Add(sinPoly, Element(DataType::DT_FP32, scoef1));
    sinPoly = Mul(xPow, sinPoly);
    sinPoly = Add(sinPoly, Element(DataType::DT_FP32, F_1));
    sinPoly = Mul(xFix, sinPoly);

    constexpr float ccoef4 = 0.0000243904487962774090654f;
    constexpr float ccoef3 = -0.00138867637746099294692f;
    constexpr float ccoef2 = 0.0416666233237390631894f;
    constexpr float ccoef1 = -0.499999997251031003120f;
    auto cosPoly = Mul(xPow, Element(DataType::DT_FP32, ccoef4));
    cosPoly = Add(cosPoly, Element(DataType::DT_FP32, ccoef3));
    cosPoly = Mul(xPow, cosPoly);
    cosPoly = Add(cosPoly, Element(DataType::DT_FP32, ccoef2));
    cosPoly = Mul(xPow, cosPoly);
    cosPoly = Add(cosPoly, Element(DataType::DT_FP32, ccoef1));
    cosPoly = Mul(xPow, cosPoly);
    cosPoly = Add(cosPoly, Element(DataType::DT_FP32, F_1));

    auto temp1 = Mul(sinPoly, ifsin);
    cosPoly = Mul(cosPoly, ifcos);
    auto res = Add(temp1, cosPoly);
    res = Mul(res, sign);
    if (dType != res.GetStorage()->Datatype()) {
        res = Cast(res, dType);
    }
    return res;
}
} // namespace npu::tile_fwk