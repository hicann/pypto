/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

inline bool IsGmGatherElement(const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_GATHER_ELEMENT || op.GetIOperands().size() != 2U) {
        return false;
    }
    const auto& inputs = op.GetIOperands();
    return inputs[0] != nullptr && inputs[1] != nullptr &&
           inputs[0]->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR &&
           inputs[1]->GetMemoryTypeToBe() == MemoryType::MEM_UB;
}

} // namespace npu::tile_fwk
