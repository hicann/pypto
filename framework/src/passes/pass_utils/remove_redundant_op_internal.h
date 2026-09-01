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
 * \file remove_redundant_op_internal.h
 * \brief Internal shared helpers for RemoveRedundantOp and RemoveRedundantOpUtils.
 */

#ifndef PASS_REMOVE_REDUNDANT_OP_INTERNAL_H_
#define PASS_REMOVE_REDUNDANT_OP_INTERNAL_H_

#include <unordered_set>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk::remove_redundant_op_internal {

std::vector<Operation*> GetTensorConsumers(const LogicalTensorPtr& tensor);
std::vector<Operation*> GetTensorProducers(const LogicalTensorPtr& tensor);
std::vector<Operation*> CollectViewAssembleCascadeOps(const LogicalTensorPtr& startTensor,
                                                      const LogicalTensorPtr& endTensor);

bool CanMigrateRemovedOpsTokenDependency(Function& function, const std::vector<Operation*>& removedOps,
                                         const std::vector<Operation*>& targetConsumers,
                                         const std::vector<Operation*>& targetProducers);

void MigrateRemovedOpsTokenDependency(Function& function, const std::vector<Operation*>& removedOps,
                                      const std::vector<Operation*>& targetConsumers,
                                      const std::vector<Operation*>& targetProducers);

bool HasOtherAssembleOutputOnSameRaw(Function& function, const LogicalTensorPtr& output,
                                     const Operation* ignoredOp = nullptr);

} // namespace npu::tile_fwk::remove_redundant_op_internal

#endif // PASS_REMOVE_REDUNDANT_OP_INTERNAL_H_
