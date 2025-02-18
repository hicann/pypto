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
 * \file runtime_slot.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace npu::tile_fwk {

template<typename TypeData, typename TypeEnum>
struct DescData {
    DescData() = default;
    DescData(TypeData v) : data_(v) {}

    void Add(TypeEnum enumValue) {
        data_ |= (1 << static_cast<int>(enumValue));
    }
    bool Count(TypeEnum enumValue) const {
        return data_ & (1 << static_cast<int>(enumValue));
    }
    DescData operator&(DescData<TypeData, TypeEnum> other) const {
        return DescData(data_ & other.data_);
    }
    DescData operator|(DescData<TypeData, TypeEnum> other) const {
        return DescData(data_ | other.data_);
    }
private:
    TypeData data_{0};
};

enum class RuntimeSlotKind : int {
    INNER,
    EXCLUSIVE_OUTCAST,
    INPUT,
    OUTPUT,
    ASSEMBLE_OUTCAST,
    INPLACE_INCAST,
    ADDRESS_EXPRESSION, // For shared memory tensor address expression
};

using RuntimeSlotKindSet = DescData<uint32_t, RuntimeSlotKind>;

struct RuntimeSlotDesc {
    RuntimeSlotKind kind;
    union {
        /* For INPLACE_INCAST */
        int inplaceIncastIndex;
        /* For ADDRESS_EXPRESSION */
        int expressionIndex;
    };
};

static_assert(sizeof(RuntimeSlotDesc) == sizeof(uint64_t), "Invalid size");

} // namespace npu::tile_fwk
