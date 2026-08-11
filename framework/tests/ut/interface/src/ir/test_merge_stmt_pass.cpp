/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

#include "tilefwk/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

using namespace npu::tile_fwk;

// Allocate a tensor via a TENSOR_ALLOC statement appended to `body`, returning the resulting
// logical tensor. Hand-built-IR helper reusable across merge-pass test cases.
LogicalTensorPtr AllocTensor(IRBuilder& builder, std::vector<ir::StmtPtr>& body, DataType dtype,
                             std::vector<int64_t> shape, std::string name)
{
    auto t = Tensor(dtype, shape, name);
    body.push_back(
        builder.CreateTensorOpStmt({t.GetStorage()}, nullptr, "TENSOR_ALLOC", {}, {}, {}, ir::Span::Unknown()));
    return t.GetStorage(false);
}

/*
@ir.function
def kernel(a@0: ir.Tensor, b@1: ir.Tensor):
    oi_update@2 = TENSOR_ALLOC()
    oi_update1@3 = TENSOR_ALLOC()
    for loop_idx_31, (oi_update_2, oi_update_3) in ir.range(0, 10, 1, init_values=(oi_update@2, oi_update1@3),
attrs=...): # ---- top-level if #1 ---- if (loop_idx_31==0): if (10<=(loop_idx_31+1)): oi_update_8, oi_update_9 =
ir.yield_(oi_update_2@2, oi_update_3@3) else: oi_update_4@2 = ADDS(a@0) oi_update_8, oi_update_9 =
ir.yield_(oi_update_4@2, oi_update_3@3) oi_update_20, oi_update_21 = ir.yield_(oi_update_8@2, oi_update_9@3) else:
            oi_update_10@2 = ADD(a@0, oi_update_2@2)
            if (10<=(loop_idx_31+1)):
                b@1 = ASSEMBLE(oi_update_10@2, attrs=["toOffset": [0, 0]])
                oi_update_14, oi_update_15 = ir.yield_(oi_update_2@2, oi_update_3@3)
            else:
                oi_update_14, oi_update_15 = ir.yield_(oi_update_10@2, oi_update_3@3)
            oi_update_20, oi_update_21 = ir.yield_(oi_update_14@2, oi_update_15@3)
        # ---- top-level if #2 ----
        if (loop_idx_31==0):
            if (10<=(loop_idx_31+1)):
                oi_update_25, oi_update_26 = ir.yield_(oi_update_20@2, oi_update_21@3)
            else:
                oi_update_22@3 = ADDS(a@0)
                oi_update_25, oi_update_26 = ir.yield_(oi_update_20@2, oi_update_22@3)
            oi_update_35, oi_update_36 = ir.yield_(oi_update_25@2, oi_update_26@3)
        else:
            oi_update_27@3 = ADD(a@0, oi_update_21@3)
            if (10<=(loop_idx_31+1)):
                b@1 = ASSEMBLE(oi_update_27@3, attrs=["toOffset": [0, 0]])
                oi_update_30, oi_update_31 = ir.yield_(oi_update_20@2, oi_update_21@3)
            else:
                oi_update_30, oi_update_31 = ir.yield_(oi_update_20@2, oi_update_27@3)
            oi_update_35, oi_update_36 = ir.yield_(oi_update_30@2, oi_update_31@3)
        # ---- loop terminator ----
        oi_update_40, oi_update_41 = continue oi_update_35@2, oi_update_36@3
    return a@0, b@1
*/
TEST(MergeStmtPass, TestMergeStmtsIntoIf)
{
    auto builder = IRBuilder();
    const auto span = ir::Span::Unknown();

    // Lift any Var/LogicalTensor shared_ptr to an ir::ExprPtr operand.
    auto AsE = [](auto v) { return std::static_pointer_cast<const ir::Expr>(v); };
    // Fresh 32x32 fp32 SSA tensor var.
    auto NewVar = [&](std::string name) {
        return builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, std::move(name));
    };
    auto Yield = [&](std::vector<ir::ExprPtr> vals) { return builder.CreateYieldStmt(std::move(vals), span); };
    auto Seq = [&](std::vector<ir::StmtPtr> stmts) { return builder.CreateSeqStmts(std::move(stmts), span); };
    auto If = [&](ir::ExprPtr cond, std::vector<ir::VarPtr> retVars, ir::StmtPtr thenBody, ir::StmtPtr elseBody) {
        return builder.CreateIfStmt(cond, thenBody, std::optional<ir::StmtPtr>{elseBody}, std::move(retVars), span);
    };

    std::vector<ir::StmtPtr> body;

    // Function inputs a, b (kept as params so the merge pass treats them as external vars).
    auto aLt = Tensor(DT_FP32, {32, 32}, "a").GetStorage(false);
    auto bLt = Tensor(DT_FP32, {32, 32}, "b").GetStorage(false);

    // oi_update@2, oi_update1@3 = TENSOR_ALLOC()
    auto oiUpdate0 = AllocTensor(builder, body, DT_FP32, {32, 32}, "oi_update");
    auto oiUpdate1 = AllocTensor(builder, body, DT_FP32, {32, 32}, "oi_update1");

    auto AddsA = [&]() {
        auto r = NewVar("adds_a");
        auto s = builder.CreateTensorOpStmt({r}, nullptr, "ADDS", {AsE(aLt)}, {}, {}, span);
        return std::make_pair(r, s);
    };
    auto AddA = [&](ir::ExprPtr x) {
        auto r = NewVar("add_a");
        auto s = builder.CreateTensorOpStmt({r}, nullptr, "ADD", {AsE(aLt), x}, {}, {}, span);
        return std::make_pair(r, s);
    };
    auto AsmB = [&](ir::ExprPtr x) {
        return builder.CreateTensorOpStmt({bLt}, nullptr, "ASSEMBLE", {x}, {},
                                          {{"toOffset", std::vector<int64_t>{0, 0}}}, span);
    };

    // Loop index symbol and the two carried iter-vars (oi_update_2, oi_update_3).
    auto i = builder.CreateScalarVar("loop_idx_31");
    auto zero = builder.CreateConstInt(0);
    auto one = builder.CreateConstInt(1);

    auto oiUpdate2 = builder.CreateVarLike("oiupdate", AsE(oiUpdate0)); // oi_update_2 -- carry slot 0
    auto oiUpdate3 = builder.CreateVarLike("oiupdate", AsE(oiUpdate1)); // oi_update_3 -- carry slot 1
    auto v0 = oiUpdate2;
    auto v1 = oiUpdate3;

    // (iterVar, initValue): the carry var is forwarded from the TENSOR_ALLOC initial value.
    auto it0 = builder.CreateIterArg(oiUpdate2, AsE(oiUpdate0));
    auto it1 = builder.CreateIterArg(oiUpdate3, AsE(oiUpdate1));

    auto start = builder.CreateConstInt(0);
    auto stop = builder.CreateScalarVar("n");
    auto step = builder.CreateConstInt(1);
    auto oiUpdate40 = builder.CreateVarLike("oiupdate", AsE(oiUpdate0)); // loop return vars
    auto oiUpdate41 = builder.CreateVarLike("oiupdate", AsE(oiUpdate1));

    // Conditions built from SymbolicScalar so the merge pass can SAT-classify branches.
    auto cond0 = i == zero;         // loop_idx == 0
    auto cond1 = (i + one) >= stop; // loop_idx + 1 >= n  (last-iteration guard)

    std::vector<ir::StmtPtr> forBodyStmts;

    // ---- top-level if #1: cond loop_idx==0, returnVars [oi_update_20, oi_update_21] ----
    auto oiUpdate8 = NewVar("oi_update_8");
    auto oiUpdate9 = NewVar("oi_update_9");
    auto [oiUpdate4, addsOi4] = AddsA();
    auto innerIf1 = If(cond1.AsExpr(), {oiUpdate8, oiUpdate9}, Yield({AsE(v0), AsE(v1)}),
                       Seq({addsOi4, Yield({AsE(oiUpdate4), AsE(v1)})}));
    auto oiUpdate20 = NewVar("oi_update_20");
    auto oiUpdate21 = NewVar("oi_update_21");
    auto [oiUpdate10, addOi10] = AddA(AsE(v0));
    auto oiUpdate14 = NewVar("oi_update_14");
    auto oiUpdate15 = NewVar("oi_update_15");
    auto innerIf2 = If(cond1.AsExpr(), {oiUpdate14, oiUpdate15},
                       Seq({AsmB(AsE(oiUpdate10)), Yield({AsE(v0), AsE(v1)})}), Yield({AsE(oiUpdate10), AsE(v1)}));
    forBodyStmts.push_back(If(cond0.AsExpr(), {oiUpdate20, oiUpdate21},
                              Seq({innerIf1, Yield({AsE(oiUpdate8), AsE(oiUpdate9)})}),
                              Seq({addOi10, innerIf2, Yield({AsE(oiUpdate14), AsE(oiUpdate15)})})));

    // ---- top-level if #2: cond loop_idx==0, returnVars [oi_update_35, oi_update_36] ----
    auto oiUpdate25 = NewVar("oi_update_25");
    auto oiUpdate26 = NewVar("oi_update_26");
    auto [oiUpdate22, addsOi22] = AddsA();
    auto innerIf3 = If(cond1.AsExpr(), {oiUpdate25, oiUpdate26}, Yield({AsE(oiUpdate20), AsE(oiUpdate21)}),
                       Seq({addsOi22, Yield({AsE(oiUpdate20), AsE(oiUpdate22)})}));
    auto oiUpdate35 = NewVar("oi_update_35");
    auto oiUpdate36 = NewVar("oi_update_36");
    auto [oiUpdate27, addOi27] = AddA(AsE(oiUpdate21));
    auto oiUpdate30 = NewVar("oi_update_30");
    auto oiUpdate31 = NewVar("oi_update_31");
    auto innerIf4 = If(cond1.AsExpr(), {oiUpdate30, oiUpdate31},
                       Seq({AsmB(AsE(oiUpdate27)), Yield({AsE(oiUpdate20), AsE(oiUpdate21)})}),
                       Yield({AsE(oiUpdate20), AsE(oiUpdate27)}));
    forBodyStmts.push_back(If(cond0.AsExpr(), {oiUpdate35, oiUpdate36},
                              Seq({innerIf3, Yield({AsE(oiUpdate25), AsE(oiUpdate26)})}),
                              Seq({addOi27, innerIf4, Yield({AsE(oiUpdate30), AsE(oiUpdate31)})})));

    // ---- loop terminator: continue(oi_update_35, oi_update_36) ----
    forBodyStmts.push_back(builder.CreateContinueStmt({AsE(oiUpdate35), AsE(oiUpdate36)}, span));

    auto forStmt = builder.CreateForStmt(i.AsVar(), start.AsExpr(), stop.AsExpr(), step.AsExpr(), {it0, it1},
                                         Seq(forBodyStmts), {oiUpdate40, oiUpdate41}, span);
    body.push_back(forStmt);
    body.push_back(builder.CreateReturnStmt({AsE(aLt), AsE(bLt)}, span));

    std::vector<ir::VarPtr> params = {std::static_pointer_cast<const ir::Var>(aLt),
                                      std::static_pointer_cast<const ir::Var>(bLt)};
    auto func = std::make_shared<ir::Function>("TestMergeStmtsIntoIf", params, std::vector<ir::TypePtr>{}, Seq(body),
                                               span);
    auto prog = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{func}, "test", span);

    // Code-coverage smoke test: run MergeStmtsIntoIf on the post-DCE if-tree and confirm the
    // loop body is still well-formed afterwards.
    auto outProg = pypto::ir::pass::MergeStmtsIntoIf()(prog);
    ASSERT_NE(outProg, nullptr);
    auto outFunc = outProg->functions_.at("TestMergeStmtsIntoIf");
    ASSERT_NE(outFunc->body_, nullptr);

    auto outSeq = std::dynamic_pointer_cast<const ir::SeqStmts>(outFunc->body_);
    ASSERT_NE(outSeq, nullptr) << "function body must remain a SeqStmts";
    bool foundFor = false;
    for (const auto& s : outSeq->stmts_) {
        if (std::dynamic_pointer_cast<const ir::ForStmt>(s)) {
            foundFor = true;
            break;
        }
    }
    EXPECT_TRUE(foundFor) << "the loop must survive MergeStmtsIntoIf";
}
