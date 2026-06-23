/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_UTILS_STMT_UTILS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_STMT_UTILS_H_

#include <memory>
#include <unordered_set>
#include <vector>

#include "ir/stmt.h"

namespace pypto {
namespace ir {
namespace utils {

/// Collect all Var references from a single expression.
std::unordered_set<const Var*> CollectVarUses(const ExprPtr& expr);

/// Collect all Var references from a single stmt.
std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt);

/// Collect all Var references from a list of stmts.
/// If skip_iter_updates, skip YieldStmt/BreakStmt/ContinueStmt
/// (their values are iter_arg updates, not uses).
std::unordered_set<const Var*> CollectStmtVarRefs(const std::vector<StmtPtr>& stmts,
                                                   bool skip_iter_updates = false);

/// Flatten a SeqStmtsPtr body to its stmts vector (returns empty if null).
const std::vector<StmtPtr>& FlattenBody(const SeqStmtsPtr& body);

/// Pointer-identity check: true if two stmt lists are identical by pointer.
bool StmtsEqual(const std::vector<StmtPtr>& a, const std::vector<StmtPtr>& b);

/// Build a SeqStmts from a vector of stmts.
SeqStmtsPtr MakeSeqBody(const std::vector<StmtPtr>& stmts, const Span& span);

}  // namespace utils
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_STMT_UTILS_H_
