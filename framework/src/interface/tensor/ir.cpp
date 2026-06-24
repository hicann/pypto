/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir.h"
#include "ir/kind_traits.h"
#include "ir/type.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/utils/dead_code_elimination.h"

#include "symbolic_scalar.h"
#include "logical_tensor.h"
#include "token_pass.h"

using npu::tile_fwk::LogicalTensor;
using npu::tile_fwk::RawSymbolicExpression;
using npu::tile_fwk::RawTensor;

namespace pypto::ir {
std::string DumpScalarExpr(const ScalarExprPtr& op)
{
    auto p = std::dynamic_pointer_cast<const RawSymbolicExpression>(op);
    ASSERT(p) << "not a RawSymbolicExpression";
    return p->Dump();
}

std::string DumpTensorVar(const VarPtr& var)
{
    auto t = std::dynamic_pointer_cast<const LogicalTensor>(var);
    ASSERT(t) << "not a logical tensor";
    return var->name_ + ": " + t->DumpType();
}

Pass pass::AggressiveDCE()
{
    return pass::CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            // Collect RawTensors from function input parameters.
            std::unordered_set<RawTensor*> rawTensors;
            for (const auto& param : func->params_) {
                if (auto type = ir::As<LogicalTensorType>(param->GetType())) {
                    auto t = std::dynamic_pointer_cast<const LogicalTensor>(param);
                    rawTensors.insert(t->tensor.get());
                }
            }

            auto isRemovable = [&rawTensors](const StmtPtr& stmt) -> bool {
                // TensorOpStmt: removable unless OP_ASSEMBLE with matching input memref
                if (auto tensorOp = std::dynamic_pointer_cast<const TensorOpStmt>(stmt)) {
                    if (tensorOp->opcode_ != "ASSEMBLE")
                        return true;
                    auto ret = tensorOp->result_[0];
                    auto tensor = std::dynamic_pointer_cast<const LogicalTensor>(ret);
                    return rawTensors.count(tensor->tensor.get()) == 0;
                }
                if (auto assignOp = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
                    return true;
                }
                if (auto scalarOp = std::dynamic_pointer_cast<const ScalarOpStmt>(stmt)) {
                    return true;
                }
                // Everything else (ReturnStmt, YieldStmt, control flow, EvalStmt): not removable
                return false;
            };

            auto newStmts = dce::EliminateDeadCode(func->body_->stmts_, isRemovable);
            if (newStmts.size() == func->body_->stmts_.size()) {
                bool changed = false;
                for (size_t i = 0; i < newStmts.size(); ++i) {
                    if (newStmts[i].get() != func->body_->stmts_[i].get()) {
                        changed = true;
                        break;
                    }
                }
                if (!changed)
                    return func;
            }
            auto newBody = std::make_shared<ir::SeqStmts>(std::move(newStmts), func->body_->span_);
            return std::make_shared<ir::Function>(
                func->name_, func->params_, func->returnTypes_, newBody, func->span_, func->funcType_);
        },
        "AggressiveDCE");
}

Pass pass::TokenPass()
{
    return pass::CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            npu::tile_fwk::TokenPass transform;
            return transform(func);
        },
        "TokenPass");
}
} // namespace pypto::ir
