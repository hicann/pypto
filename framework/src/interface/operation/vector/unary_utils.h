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
 * \file unary_utils.h
 * \brief
 */

#pragma once

#include "unary.h"

namespace npu::tile_fwk {

template <typename T>
int64_t MultiplyLastTwoDims(const std::vector<int64_t>& vec)
{
    constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
    constexpr size_t ELEMENT_SIZE = sizeof(T);
    constexpr size_t ALIGN_ELEMENTS = ALIGN_SIZE / ELEMENT_SIZE;
    int64_t axis2 = (vec[vec.size() - 1] + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
    return axis2 * vec[vec.size() - NUM_VALUE_2];
}

int64_t CmpResAlign(const std::vector<int64_t>& vec);

} // namespace npu::tile_fwk
