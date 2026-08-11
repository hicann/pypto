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
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

namespace {
static constexpr int64_t TILE = 16;

ir::Span Sp() { return ir::Span("test_canonicalize_pass", 1, 1); }

struct IrFuncSetup {
    npu::tile_fwk::IRBuilder builder;
    std::shared_ptr<npu::tile_fwk::Function> fwkFunc;
    LogicalTensors params;
    std::vector<ir::StmtPtr> stmts;

    explicit IrFuncSetup(const std::string& name)
    {
        fwkFunc = std::make_shared<npu::tile_fwk::Function>(Program::GetInstance(), name + "_magic", name, nullptr);
        fwkFunc->SetFunctionType(FunctionType::DYNAMIC);
        fwkFunc->SetGraphType(GraphType::TENSOR_GRAPH);
        Program::GetInstance().InsertFuncToFunctionMap(fwkFunc->GetMagicName(), fwkFunc);
        Program::GetInstance().SetCurrentFunction(fwkFunc.get());
    }

    LogicalTensorPtr MakeParam(const std::string& name)
    {
        auto lt = builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, name);
        params.push_back(lt);
        return lt;
    }

    ir::FunctionPtr BuildIrFunction(const std::string& name)
    {
        auto body = std::make_shared<ir::SeqStmts>(stmts, Sp());
        std::vector<ir::VarPtr> irParams;
        for (auto& p : params) {
            irParams.push_back(std::static_pointer_cast<const ir::Var>(p));
        }
        return std::make_shared<ir::Function>(name, irParams, std::vector<ir::TypePtr>{}, body, Sp());
    }
};

// Lift any Var/Const/LogicalTensor shared_ptr to an ir::ExprPtr.
auto AsExpr = [](auto v) { return std::static_pointer_cast<const ir::Expr>(v); };
} // namespace

// Mirrors of python/tests/ut/ir/test_merge_pass/test_oi_update1.py
TEST(CanonicalizePassTest, OiUpdate1)
{
    IrFuncSetup setup("OiUpdate1");
    auto a = setup.MakeParam("a");
    auto b = setup.MakeParam("b");

    auto& builder = setup.builder;
    const auto span = Sp();
    const auto fp32 = a->GetType();

    // Initial values for the two carries (the TENSOR_ALLOC inputs oi_update/oi_update1).
    auto oiUpdateInit = builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, "oi_update_init");
    auto oiUpdate1Init = builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, "oi_update1_init");

    // for loop_idx in range(0, 10, 1)
    auto loopVar = std::make_shared<ir::Var>("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), span);
    auto zero = std::make_shared<ir::ConstInt>(0, ir::DataType::INT64, span);
    auto ten = std::make_shared<ir::ConstInt>(10, ir::DataType::INT64, span);
    auto one = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, span);
    auto boolType = std::make_shared<ir::ScalarType>(ir::DataType::BOOL);

    // Two carries. v1 (oi_update1) is read only inside the inner if-yield below.
    auto carry0 = builder.CreateIterArg("oi_update", fp32, AsExpr(oiUpdateInit), span);
    auto carry1 = builder.CreateIterArg("oi_update1", fp32, AsExpr(oiUpdate1Init), span);
    auto v0 = carry0->iterVar_; // oi_update
    auto v1 = carry1->iterVar_; // oi_update1

    std::vector<ir::StmtPtr> bodyStmts;

    // tmp = ADD(a, oi_update)  -- carry0 directly consumed, always live.
    auto tmp = builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, "tmp");
    bodyStmts.push_back(builder.CreateTensorOpStmt(
        {tmp}, nullptr, "ADD", std::vector<ir::ExprPtr>{AsExpr(a), AsExpr(v0)}, std::vector<ir::VarPtr>{}, {}, span));
    // b = ASSEMBLE(tmp)  -- side effect on input b.
    bodyStmts.push_back(builder.CreateTensorOpStmt({b}, nullptr, "ASSEMBLE", std::vector<ir::ExprPtr>{AsExpr(tmp)},
                                                   std::vector<ir::VarPtr>{}, {}, span));

    // fwd1 = if (loop_idx == 0): yield(oi_update1) else: yield(oi_update1)
    //   The ONLY use of carry1's iter-var is inside this yield -- the pre-fix
    //   collector skipped it and dropped the carry.
    auto fwd1 = builder.CreateTensorVar(DT_FP32, {32, 32}, TileOpFormat::TILEOP_ND, "fwd1");
    auto cond = std::make_shared<ir::Call>("==", std::vector<ir::ExprPtr>{AsExpr(loopVar), AsExpr(zero)}, boolType,
                                           span);
    auto ifStmt = builder.CreateIfStmt(
        cond, builder.CreateYieldStmt(std::vector<ir::ExprPtr>{AsExpr(v1)}, span),
        std::optional<ir::StmtPtr>{builder.CreateYieldStmt(std::vector<ir::ExprPtr>{AsExpr(v1)}, span)},
        std::vector<ir::VarPtr>{fwd1}, span);
    bodyStmts.push_back(ifStmt);

    // b = ASSEMBLE(fwd1)  -- side effect keeps fwd1 alive, and via the if-yield keeps oi_update1 alive.
    bodyStmts.push_back(builder.CreateTensorOpStmt({b}, nullptr, "ASSEMBLE", std::vector<ir::ExprPtr>{AsExpr(fwd1)},
                                                   std::vector<ir::VarPtr>{}, {}, span));

    // Loop terminator: forward the new carried values.
    bodyStmts.push_back(builder.CreateYieldStmt(std::vector<ir::ExprPtr>{AsExpr(tmp), AsExpr(fwd1)}, span));

    auto body = builder.CreateSeqStmts(bodyStmts, span);

    // Final values of the carries are unused after the loop (only a, b are returned).
    auto rv0 = IRContext::Get().MakeVar("oi_update_rv", fp32, span);
    auto rv1 = IRContext::Get().MakeVar("oi_update1_rv", fp32, span);
    auto forStmt = builder.CreateForStmt(loopVar, AsExpr(zero), AsExpr(ten), AsExpr(one),
                                         std::vector<ir::IterArgPtr>{carry0, carry1}, body,
                                         std::vector<ir::VarPtr>{rv0, rv1}, span);
    setup.stmts.push_back(forStmt);
    setup.stmts.push_back(builder.CreateReturnStmt(std::vector<ir::ExprPtr>{AsExpr(a), AsExpr(b)}, span));

    auto irFunc = setup.BuildIrFunction("OiUpdate1");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto outProg = pypto::ir::pass::Canonicalize()(irProg);
    auto& outFunc = outProg->functions_.at("OiUpdate1");
    const auto& stmts = outFunc->body_->stmts_;

    auto forStmtOut = std::dynamic_pointer_cast<const ir::ForStmt>(stmts[0]);
    ASSERT_NE(forStmtOut, nullptr) << "first body stmt must remain the loop";

    // Both carries must survive canonicalize. The pre-fix code dropped `oi_update1`
    // (its iter-var was read only inside a yield) -- see test_oi_update1.py.
    ASSERT_EQ(forStmtOut->iterArgs_.size(), 2u);
    ASSERT_EQ(forStmtOut->returnVars_.size(), 2u);
    EXPECT_EQ(forStmtOut->iterArgs_[0]->iterVar_->name_, "oi_update");
    EXPECT_EQ(forStmtOut->iterArgs_[1]->iterVar_->name_, "oi_update1");
}
