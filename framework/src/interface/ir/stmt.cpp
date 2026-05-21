/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/stmt.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/core.h"
#include "ir/span.h"

namespace pypto {
namespace ir {

OpStmts::OpStmts(std::vector<StmtPtr> stmts, Span span) : Stmt(std::move(span)), stmts_(std::move(stmts))
{
    // Validate that all statements are AssignStmt or EvalStmt
    for (size_t i = 0; i < stmts_.size(); ++i) {
        const auto& stmt = stmts_[i];
        INTERNAL_CHECK(stmt) << "OpStmts has null statement at index " << i;
        auto kind = stmt->GetKind();
        INTERNAL_CHECK(kind == ObjectKind::AssignStmt || kind == ObjectKind::EvalStmt)
            << "OpStmts only accepts AssignStmt or EvalStmt, but got " << stmt->TypeName() << " at index " << i;
    }
}

} // namespace ir
} // namespace pypto
