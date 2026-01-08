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
 * \file runtime_outcast_tensor.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include <sstream>

using uintdevptr_t = uint64_t;
using intdevptr_t = int64_t;

namespace npu::tile_fwk::dynamic {

#define X(value) value,
enum class RuntimeTensorMemProperty : uint8_t {
#include "machine/utils/dynamic/runtime_tensor_mem_properties.in"
};
#undef X

inline constexpr const char *GetRuntimeTensorMemPropertyName(RuntimeTensorMemProperty property) {
#define X(value) #value,
    constexpr const char *NAMELIST[] = {
#include "machine/utils/dynamic/runtime_tensor_mem_properties.in"
    };
#undef X

    return NAMELIST[static_cast<size_t>(property)];
}

struct RuntimeOutcastTensor {
    uintdevptr_t addr;
    RuntimeTensorMemProperty property;
    uint32_t refCnt;

    RuntimeOutcastTensor(uintdevptr_t taddr, RuntimeTensorMemProperty tproperty, uint32_t trefCnt)
        : addr(taddr), property(tproperty), refCnt(trefCnt) {}

    std::string Dump() const {
        std::stringstream ss;
        ss << "&0x" << std::hex << addr << ", " << GetRuntimeTensorMemPropertyName(property);
        return std::move(ss).str();
    }
};

} // namespace npu::tile_fwk::dynamic