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
 * \file calc_error.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk::calc_error {

// Calculator 层错误码从 0xBF000U 开始，只在 calculator/ 目录内部使用，

enum class CalculatorErrorScene : uint32_t {
    // Range / 生成相关
    RANGE_NUMEL_MISMATCH            = 0xBF000U, // torch::arange 生成的 numel 与 out.shape 展开后的元素个数不一致

    // 比较运算相关
    COMPARE_UNSUPPORTED_TYPE        = 0xBF001U, // CompareImpl 收到未支持的 CmpOperationType
    BITMODE_LAST_DIM_INVALID        = 0xBF002U, // CmpModeType::BIT 模式下，最后一维尺寸不是 NUM_VALUE_8 的倍数

    // 形状约束 / 格式转换
    FORMAT_ND2NZ_RANK_LT_2          = 0xBF003U, // FormatND2NZ 要求 rank >= 2，实际小于 2
    FORMAT_NZ2ND_RANK_LT_2          = 0xBF004U, // FormatNZ2ND 要求 rank >= 2，实际小于 2

    // 量化预处理相关
    QUANTPRECOMPUTE_NULL_DATAPTR    = 0xBF005U, // QuantPreCompute 中 out/self 的 dataPtr 为空
    QUANTPRECOMPUTE_DTYPE_MISMATCH  = 0xBF006U, // QuantPreCompute 要求 out=FP16/self=INT32 的 dtype 组合不满足

    // GatherMask / pattern 模式
    GATHERMASK_PATTERNMODE_INVALID  = 0xBF007U, // GatherMask 中 patternMode 不在 [1,7] 合法范围内

    // TopK/MrgSort 轴约束
    MRGSORT_AXIS_OUT_OF_RANGE       = 0xBF008U, // MrgSort/TiledMrgSort 等中 axis 超出张量维度范围

    // Scatter/ScatterUpdate 相关
    SCATTER_BLOCKSIZE_ZERO          = 0xBF009U, // ScatterUpdate 中 blockSize 为 0
    SCATTER_INDICES_DIM_INVALID     = 0xBF00AU, // indices 维度不是期望的 2 维
    SCATTER_SRC_RET_DIM_UNSUPPORTED = 0xBF00BU, // src/ret 的维度不是 2 或 4，当前实现不支持
    SCATTER_SRC_RET_DIM_MISMATCH    = 0xBF00CU, // src 与 ret 的维度数量不一致
};

} // namespace npu::tile_fwk::calc_error


