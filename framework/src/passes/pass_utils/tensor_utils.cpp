/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_utils.cpp
 * \brief utils for querying logical tensors sharing the same rawMagic.
 */

#include "tensor_utils.h"
#include <set>
#include "interface/operation/operation.h"

namespace npu {
namespace tile_fwk {

std::vector<LogicalTensorPtr> TensorUtils::GetSameRawMagicLogicalTensors(Function& function,
                                                                         const LogicalTensorPtr& tensor)
{
    std::vector<LogicalTensorPtr> result;
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return result;
    }
    auto candidates = GraphUtils::GetTensorsByRawMagic(function, tensor->tensor->rawmagic);
    result.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate == nullptr) {
            continue;
        }
        result.push_back(candidate);
    }
    return result;
}

std::vector<Operation*> TensorUtils::GetProducersOfSameRawMagicLogicalTensors(Function& function,
                                                                              const LogicalTensorPtr& tensor)
{
    std::vector<Operation*> result;
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return result;
    }
    std::set<Operation*> seen;
    for (const auto& t : GetSameRawMagicLogicalTensors(function, tensor)) {
        if (t == nullptr) {
            continue;
        }
        for (auto* producer : t->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted() || producer->BelongTo() != &function) {
                continue;
            }
            if (seen.insert(producer).second) {
                result.push_back(producer);
            }
        }
    }
    return result;
}
} // namespace tile_fwk
} // namespace npu
