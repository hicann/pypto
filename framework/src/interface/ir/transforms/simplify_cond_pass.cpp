/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */
#include <functional>
#include <utility>
#include <vector>

#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"

#include "tilefwk/symbolic_scalar.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"

namespace pypto::ir {

namespace {

using npu::tile_fwk::AsLogicalTensor;
using npu::tile_fwk::Operation;
using npu::tile_fwk::RawSymbolicScalarPtr;
using npu::tile_fwk::SymbolicScalar;

class CondSimplifier : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    void VisitStmt_(const IfStmtPtr& op) override
    {
        auto cond = SymbolicScalar::FromExpr(op->condition_);

        conds_.push_back({cond.Raw(), true_.Raw()});
        VisitStmt_(op->thenBody_);
        conds_.pop_back();

        if (op->elseBody_) {
            conds_.push_back({cond.Raw(), false_.Raw()});
            VisitStmt_(*op->elseBody_);
            conds_.pop_back();
        }
    }

    void VisitExpr_(const VarPtr& var) override
    {
        if (auto lt = AsLogicalTensor(var)) {
            for (auto& s : lt->GetDynValidShape()) {
                s = s.Substitute(conds_).Simplify();
            }
            return;
        }
    }

    void VisitStmt_(const TensorOpStmtPtr& op) override
    {
        for (auto& arg : op->args_) {
            VisitExpr(arg);
        }
        for (auto& res : op->result_) {
            VisitExpr(res);
        }
        auto oper = std::dynamic_pointer_cast<Operation>(std::const_pointer_cast<TensorOpStmt>(op));
        if (oper) {
            for (auto& attr : oper->GetDynamicAttributeList()) {
                auto& s = attr.get();
                s = s.Substitute(conds_).Simplify();
            }
            return;
        }
    }

    std::vector<std::pair<RawSymbolicScalarPtr, RawSymbolicScalarPtr>> conds_;
    SymbolicScalar true_{1};
    SymbolicScalar false_{0};
};

} // namespace

namespace pass {

Pass SimplifySymbolicScalar()
{
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (!func || !func->body_) {
                return func;
            }
            CondSimplifier rewriter;
            rewriter.VisitStmt(func->body_);
            return func;
        },
        "SimplifySymbolicScalar");
}

} // namespace pass

} // namespace pypto::ir
