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
 * \file conv_pto.h
 * \brief Conv TileOp Interface Definition
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_PTO__H
#define TILEOP_TILE_OPERATOR_CONV_PTO__H

// Common Operator Definitions (Shared by Atlas A3, Ascend 950PR/Ascend 950DT)
#include "impl/conv_utils.h"
#include "impl/conv_load2d_load3d_impl.h"
#include "impl/conv_load_nz2nz_impl.h"
#include "impl/conv_store_nz2nz_impl.h"

// Operator Implementation for Ascend 950PR/Ascend 950DT Architectures
#include "impl/arch35/conv_load_dn2nz_impl.h"
#include "impl/arch35/conv_store_nz2dn_impl.h"

// Copy data from DDR to L1
template <CopyInMode mode, bool isConv3D, bool isFmap, typename T, typename U>
TILEOP void TLoadConv(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                      const int64_t& offset3, const int64_t& offset4, const int64_t& shape0, const int64_t& shape1,
                      const int64_t& shape2, const int64_t& shape3, const int64_t& shape4)
{
    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
                  "[TLoadConv Error]: Src format shoulde be GM and Dst format shoulde be L1");
    OffsetInfo offsetInfo = {offset0, offset1, offset2, offset3, offset4};
    ShapeInfo srcShapeInfo = {shape0, shape1, shape2, shape3, shape4};
    if constexpr (mode == CopyInMode::ND2NZ) {
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        TLoadConvDN2NZ<isConv3D, isFmap>(dst, src, offsetInfo, srcShapeInfo);
    } else if constexpr (mode == CopyInMode::NZ2NZ) {
        TLoadConvNZ2NZ<isConv3D, isFmap>(dst, src, offsetInfo, srcShapeInfo);
    }
}

// Copy data from L0C to DDR
template <CopyOutMode mode, bool isConv3D, int64_t reluType, typename T, typename U>
TILEOP void TStoreConv(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                       const int64_t& offset3, const int64_t& offset4, const int64_t& realM, const int64_t& realN,
                       const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(srcShapeSize == SHAPE_DIM2, "L0C shape size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::GM && U::FORMAT == Hardware::L0C,
                  "[TStoreConv Error]: Src format shoulde be L0C and Dst format shoulde be GM");
    OffsetInfo offsetInfo = {offset0, offset1, offset2, offset3, offset4};
    if constexpr (mode == CopyOutMode::NZ2ND) {
    } else if constexpr (mode == CopyOutMode::NZ2DN) {
        TStoreConvNZ2DN<isConv3D, reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    } else if constexpr (mode == CopyOutMode::NZ2NZ) {
        TStoreConvNZ2NZ<isConv3D, reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    }
}

#if defined(PTO_NPU_ARCH_A5)
// Copy data from L0C to UB, shape/offset 语义与 TStoreConv 一致，仅目的端为 UB (仅支持 A5)
// srcOffsetM/srcOffsetN: L0C 源侧 (m, n) 偏移 (VIEW 大搬小时 view 窗口在 L0C 块内的起始行列)
// subBlockId: 搬运目标 UB 窗口 (0=AIV0, 1=AIV1)，按消费链所在 AIV 核选择
template <CopyOutMode mode, bool isConv3D, int64_t reluType, typename T, typename U>
TILEOP void TCopyL0C2UBConv(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                            const int64_t& offset3, const int64_t& offset4, const int64_t& realM, const int64_t& realN,
                            const int64_t& realCutW, const int64_t& cutW, const int64_t& srcOffsetM,
                            const int64_t& srcOffsetN, const int64_t& subBlockId)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(srcShapeSize == SHAPE_DIM2, "L0C shape size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::UB && U::FORMAT == Hardware::L0C,
                  "[TCopyL0C2UBConv Error]: Src format shoulde be L0C and Dst format shoulde be UB");
    OffsetInfo offsetInfo = {offset0, offset1, offset2, offset3, offset4};
    if constexpr (mode == CopyOutMode::NZ2DN) {
        TCopyL0C2UBConvNZ2DN<isConv3D, reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW, srcOffsetM,
                                                 srcOffsetN, subBlockId);
    } else {
        static_assert(mode == CopyOutMode::NZ2DN, "[TCopyL0C2UBConv Error]: L0C to UB only support CopyOutMode::NZ2DN");
    }
}
#endif // defined(PTO_NPU_ARCH_A5)

#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H
