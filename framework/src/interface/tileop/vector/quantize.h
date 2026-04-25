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
 * \file quantize.h
 * \brief INT8 对称/非对称量化 Tile 算子
 *
 * - INT8_SYM:  FP32 -> INT8,  范围 [-128, 127]
 * - INT8_ASYM: FP32 -> UINT8, 范围 [0, 255]
 *
 * 只支持逐行量化 (axis=-1)，逐列量化在 Operation 层通过 Transpose 实现
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

/// Tile 内存对齐字节数
constexpr size_t TILE_ALIGNMENT_BYTES = 32;

/// 向上取整到对齐边界
#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

// =============================================================================
// INT8 对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8 对称量化 (逐行)
 * @param dst   输出 INT8 张量, 形状 [..., H, W]
 * @param src   输入 FP32 张量, 形状 [..., H, W]
 * @param scale 缩放因子, 形状 [..., H]
 */
template <typename T0, typename T1, typename T2>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale) {
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // 获取形状
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    // 获取 Tile 形状并计算对齐
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst,
                                    pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src,
                                    pto::BLayout::RowMajor, -1, -1>;
    using ParaTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1,
                                    pto::BLayout::ColMajor, -1, -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ParaTileDefine scaleTile(srcShape3, 1);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                // scaleOffset should need to be shifted
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));

                pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile);
            }
        }
    }
}

// =============================================================================
// INT8 非对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8 非对称量化 (逐行)
 * @param dst    输出 UINT8 张量, 形状 [..., H, W]
 * @param src    输入 FP32 张量, 形状 [..., H, W]
 * @param scale  缩放因子, 形状 [..., H]
 * @param offset 零点偏移, 形状 [..., H]
 */
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset) {
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();
    const auto offsetLayout = offset.GetLayout();

    // 获取形状
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    auto offsetStride1 = offsetLayout.template GetStrideDim<1, expectSize>();
    auto offsetStride2 = offsetLayout.template GetStrideDim<2, expectSize>();
    auto offsetStride3 = offsetLayout.template GetStrideDim<3, expectSize>();

    // 获取 Tile 形状
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr auto offsetTileH = TileOp::GetTensorTileShapeDim<T3, 3, expectSize>();
    constexpr int paddedRow_offset = PTO_CEIL(offsetTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;
    using OffsetDtype = typename T3::Type;

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW,
                                    pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW,
                                    pto::BLayout::RowMajor, -1, -1>;
    using ParaTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1,
                                    pto::BLayout::ColMajor, -1, -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ParaTileDefine scaleTile(srcShape3, 1);
                ParaTileDefine offsetTile(srcShape3, 1);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                // scaleOffset & offsetOffset should need to be shifted
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;
                auto offsetOffset = n0Index * offsetStride1 + n1Index * offsetStride2 + n2Index * offsetStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile);
            }
        }
    }
}

// =============================================================================
// 统一量化接口
// =============================================================================
#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief 统一量化接口
 * @tparam quantType INT8_SYM 或 INT8_ASYM
 */
template <pto::QuantType quantType, typename T0, typename T1, typename T2>
TILEOP void TQuant(T0 dst, T1 src, T2 scale) {
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant with 3 parameters only supports INT8_SYM. "
                  "TQuant only supports INT8_SYM(3 parameters) and INT8_ASYM(4 parameters).");
    TQuantInt8Sym(dst, src, scale);
}

template <pto::QuantType quantType, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset) {
    static_assert(quantType == pto::QuantType::INT8_ASYM,
                  "TQuant with 4 parameters only supports INT8_ASYM."
                  "TQuant only supports INT8_SYM(3 parameters) and INT8_ASYM(4 parameters).");
    TQuantInt8Asym(dst, src, scale, offset);
}

#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H
