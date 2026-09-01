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
 * \file pass_token_utils.h
 * \brief Token dependency utilities for passes.
 */

#ifndef PASS_TOKEN_UTILS_H
#define PASS_TOKEN_UTILS_H

#include <unordered_set>
#include <vector>
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu::tile_fwk {

class PassTokenUtils {
public:
    static void AddTokenIfAbsent(Operation& op, const ir::VarPtr& token);
    static void MoveTokenDependencyBeforeRemoveOp(Function& function, Operation& op);
    static void CopyTokenDependency(Function& function, Operation& originOp, Operation& copiedOp);
    static void AddTokenConsumer(Function& function, const ir::VarPtr& token, Operation& consumerOp);
    static void MoveResultTokensToProducers(Function& function, const std::vector<Operation*>& sourceOps,
                                            const std::vector<Operation*>& targetProducerOps,
                                            const std::unordered_set<ir::StmtPtr>& skippedConsumers);
    static void CleanupDeletedTokenDependency(Function& function, const std::vector<Operation*>& deletedOps);
};

} // namespace npu::tile_fwk

#endif // PASS_TOKEN_UTILS_H
