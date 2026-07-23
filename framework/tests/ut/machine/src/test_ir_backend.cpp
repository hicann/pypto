/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ir_backend.cpp
 * \brief Unit tests for SCF IR based control flow building functions (ir_backend.cpp).
 */

#include "gtest/gtest.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/utils/id_gen.h"
#include "machine/host/backend.h"
#include "machine/host/ir_backend.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {
ir::Span Sp() { return ir::Span("test_ir_backend", 1, 1); }

ir::ExprPtr MakeSymbolExpr(const std::string& name) { return std::make_shared<RawSymbolicSymbol>(name); }

ir::ExprPtr MakeImmediateExpr(int64_t val) { return std::make_shared<RawSymbolicImmediate>(val); }

ir::VarPtr MakeVar(const std::string& name)
{
    return std::make_shared<ir::Var>(name, std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
}

ir::StmtPtr MakeTensorOpStmt(const std::string& opcode, const std::vector<std::pair<std::string, std::any>>& attrs = {})
{
    return std::make_shared<ir::TensorOpStmt>(std::vector<ir::VarPtr>{}, nullptr, opcode, std::vector<ir::ExprPtr>{},
                                              std::vector<ir::VarPtr>{}, attrs, Sp());
}

ir::StmtPtr MakeForStmt(ir::VarPtr loopVar, ir::ExprPtr start, ir::ExprPtr stop, ir::ExprPtr step, ir::StmtPtr body,
                        const std::vector<std::pair<std::string, std::any>>& attrs = {})
{
    return std::make_shared<ir::ForStmt>(loopVar, start, stop, step, std::vector<ir::IterArgPtr>{}, body,
                                         std::vector<ir::VarPtr>{}, Sp(), attrs);
}

ir::StmtPtr MakeIfStmt(ir::ExprPtr cond, ir::StmtPtr thenBody, std::optional<ir::StmtPtr> elseBody = std::nullopt)
{
    return std::make_shared<ir::IfStmt>(cond, thenBody, elseBody, std::vector<ir::VarPtr>{}, Sp());
}

ir::StmtPtr MakeSeqStmts(std::vector<ir::StmtPtr> stmts)
{
    auto seq = std::make_shared<ir::SeqStmts>(std::move(stmts), Sp());
    return std::static_pointer_cast<const ir::Stmt>(seq);
}

struct DynFuncFixture {
    std::shared_ptr<Function> dynFunc;
    std::shared_ptr<DyndevFunctionAttribute> dynAttr;

    DynFuncFixture(const std::string& name = "test_dyn")
    {
        Program::GetInstance().Reset();
        dynFunc = std::make_shared<Function>(Program::GetInstance(), name + "_magic", name, nullptr);
        dynFunc->SetFunctionType(FunctionType::DYNAMIC);
        dynFunc->SetGraphType(GraphType::TENSOR_GRAPH);
        dynAttr = std::make_shared<DyndevFunctionAttribute>();
        dynFunc->SetDyndevAttribute(dynAttr);
        Program::GetInstance().InsertFuncToFunctionMap(dynFunc->GetMagicName(), dynFunc);
        Program::GetInstance().SetCurrentDynamicFunction(dynFunc.get());
    }

    ~DynFuncFixture()
    {
        Program::GetInstance().SetCurrentDynamicFunction(nullptr);
        Program::GetInstance().Reset();
    }
};

struct LinkerFixture {
    SymbolicSymbolTable symbolTable;
    DyndevFunctionAttribute::FunctionGroup funcGroup;
    DyndevFunctionAttribute::ExpressionTableDictGroup exprTableDictGroup;
    std::unique_ptr<Linker> linker;

    LinkerFixture() { linker = std::make_unique<Linker>(symbolTable, funcGroup, exprTableDictGroup); }
};
} // namespace

class TestSuite_IrBackend : public testing::Test {
protected:
    void SetUp() override { config::Reset(); }
    void TearDown() override { config::Reset(); }
};

struct ControlFlowCtx {
    IrBackendContext irBackendCtx;
    FunctionCache cache;
    std::unordered_map<int, int> slotIdxMapping;
    DyndevFunctionAttribute::FunctionGroup group;
    std::unordered_map<Function*, Function*> rootTileDict;
    std::ostringstream controlFlowOss;
    std::ostringstream expressionOss;
    std::ostringstream exprHeaderOss;
    std::vector<std::string> exprSrcFiles;
    ValDependTensorMeta meta;
};

static ir::ForStmtPtr AsForStmt(const ir::StmtPtr& stmt) { return std::dynamic_pointer_cast<const ir::ForStmt>(stmt); }

TEST_F(TestSuite_IrBackend, ExprPtrToSymbolicScalar_AllCases)
{
    EXPECT_FALSE(ExprPtrToSymbolicScalar(nullptr).IsValid());

    auto sym = ExprPtrToSymbolicScalar(MakeSymbolExpr("test_var"));
    EXPECT_TRUE(sym.IsValid());
    EXPECT_TRUE(sym.Raw()->IsSymbol());
    EXPECT_EQ(sym.Raw()->GetSymbolName(), "test_var");

    auto imm = ExprPtrToSymbolicScalar(MakeImmediateExpr(42));
    EXPECT_TRUE(imm.IsValid());
    EXPECT_TRUE(imm.Raw()->IsImmediate());

    EXPECT_THROW(ExprPtrToSymbolicScalar(MakeVar("plain_var")), npu::tile_fwk::Error);
}

TEST_F(TestSuite_IrBackend, GetLoopVarOriginName_AllCases)
{
    auto irVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    EXPECT_EQ(GetLoopVarOriginName(irVar), "loop_idx");

    auto dupVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    EXPECT_EQ(GetLoopVarOriginName(dupVar), "loop_idx");
}

TEST_F(TestSuite_IrBackend, IsOpCallStmt_AllCases)
{
    EXPECT_FALSE(IsOpCallStmt(MakeSeqStmts({})));
    EXPECT_FALSE(IsOpCallStmt(MakeTensorOpStmt("CALL")));
    EXPECT_TRUE(IsOpCallStmt(MakeTensorOpStmt("CALL", {{"callee", std::string("some_func")}})));
}

TEST_F(TestSuite_IrBackend, ResolveCalleeFromOpCall_AllCases)
{
    EXPECT_EQ(ResolveCalleeFromOpCall(MakeSeqStmts({})), nullptr);
    EXPECT_EQ(ResolveCalleeFromOpCall(MakeTensorOpStmt("CALL")), nullptr);

    DynFuncFixture fixture;
    auto calleeFunc = std::make_shared<Function>(Program::GetInstance(), "callee_magic", "callee_name", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("callee_magic", calleeFunc);
    auto result = ResolveCalleeFromOpCall(MakeTensorOpStmt("CALL", {{"callee", std::string("callee_magic")}}));
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(result->GetRawName(), "callee_name");

    EXPECT_EQ(ResolveCalleeFromOpCall(MakeTensorOpStmt("CALL", {{"callee", std::string("nonexistent_func")}})),
              nullptr);
}

TEST_F(TestSuite_IrBackend, IrBuildVirtualLoopFunc_AllCases)
{
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        fixture.dynFunc->SetDyndevAttribute(nullptr);
        EXPECT_EQ(IrBuildVirtualLoopFunc(irBackendCtx, nullptr, fixture.dynFunc.get()), fixture.dynFunc.get());
    }
    auto mkFor = [](const std::vector<std::pair<std::string, std::any>>& attrs = {}) {
        auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64),
                                                Sp());
        return MakeForStmt(loopVar, MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1), MakeSeqStmts({}),
                           attrs);
    };
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto forStmt = mkFor();
        auto* result = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_NE(result, nullptr);
        EXPECT_NE(result, fixture.dynFunc.get());
        EXPECT_EQ(result->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
        EXPECT_NE(result->GetDynloopAttribute(), nullptr);

        auto* cached = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_EQ(result, cached);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto forStmt = mkFor({{"parallel", true}});
        auto* result = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_NE(result, nullptr);
        EXPECT_TRUE(result->GetDynloopAttribute()->parallel);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto forStmt = mkFor({{"submit_before_loop", true}});
        auto* result = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        EXPECT_NE(result, nullptr);
        EXPECT_TRUE(result->GetDynloopAttribute()->submitBeforeLoop);
    }
}

TEST_F(TestSuite_IrBackend, IrParseValueDependDesc_AllCases)
{
    {
        DynFuncFixture fixture;
        fixture.dynFunc->SetDyndevAttribute(nullptr);
        EXPECT_NO_THROW(IrParseValueDependDesc(fixture.dynFunc.get(), {MakeImmediateExpr(0)}));
    }
    {
        DynFuncFixture fixture;
        EXPECT_NO_THROW(IrParseValueDependDesc(fixture.dynFunc.get(),
                                               {MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1)}));
    }
}

TEST_F(TestSuite_IrBackend, InsertCacheStopForContrlFlow_AllCases)
{
    auto mkFor = []() {
        return MakeForStmt(MakeVar("loop_idx"), MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1),
                           MakeSeqStmts({}));
    };
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        fixture.dynFunc->SetDyndevAttribute(nullptr);
        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertCacheStopForContrlFlow(irBackendCtx, AsForStmt(mkFor()).get(), fixture.dynFunc.get(), 0, oss, meta);
        EXPECT_TRUE(oss.str().empty());
        EXPECT_FALSE(meta.disableCtrlFlowCache);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertCacheStopForContrlFlow(irBackendCtx, AsForStmt(mkFor()).get(), fixture.dynFunc.get(), 0, oss, meta);
        EXPECT_TRUE(oss.str().empty());
        EXPECT_FALSE(meta.disableCtrlFlowCache);
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture fixture;
        auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64),
                                                Sp());
        auto forStmt = MakeForStmt(loopVar, MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1),
                                   MakeSeqStmts({}));
        auto* loopFunc = IrBuildVirtualLoopFunc(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get());
        auto& desc = fixture.dynAttr->valueDependDescDict[loopFunc];
        desc.getInputDataCount = 1;
        desc.getTensorDataCount = 0;

        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertCacheStopForContrlFlow(irBackendCtx, AsForStmt(forStmt).get(), fixture.dynFunc.get(), 1, oss, meta);
        EXPECT_FALSE(oss.str().empty());
        EXPECT_TRUE(oss.str().find("RUNTIME_FUNCKEY_CACHESTOP") != std::string::npos);
        EXPECT_TRUE(meta.disableCtrlFlowCache);
    }
}

TEST_F(TestSuite_IrBackend, InsertWaitAicoreStartForControlFlow_AllCases)
{
    {
        auto forStmt = MakeForStmt(MakeVar("loop_idx"), MakeImmediateExpr(0), MakeImmediateExpr(10),
                                   MakeImmediateExpr(1), MakeSeqStmts({}));
        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertWaitAicoreStartForControlFlow(AsForStmt(forStmt).get(), 1, oss, meta);
        EXPECT_TRUE(oss.str().empty());
    }
    {
        auto forStmt = MakeForStmt(MakeVar("loop_idx"), nullptr, nullptr, nullptr, MakeSeqStmts({}));
        std::ostringstream oss;
        ValDependTensorMeta meta;
        InsertWaitAicoreStartForControlFlow(AsForStmt(forStmt).get(), 1, oss, meta);
        EXPECT_TRUE(oss.str().empty());
    }
}

TEST_F(TestSuite_IrBackend, FindExprFromIRStmt_AllCases)
{
    IrBackendContext irBackendCtx;
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    FunctionCache cache;
    std::vector<ir::ExprPtr> condStack;

    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, nullptr, dynFixture.dynFunc.get(), condStack));
    EXPECT_NO_THROW(FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, MakeSeqStmts({}),
                                       dynFixture.dynFunc.get(), condStack));

    auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    auto forStmt = MakeForStmt(loopVar, MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1),
                               MakeSeqStmts({}));
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, forStmt, dynFixture.dynFunc.get(), condStack));

    auto ifBoth = MakeIfStmt(MakeSymbolExpr("cond_var"), MakeSeqStmts({}), MakeSeqStmts({}));
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifBoth, dynFixture.dynFunc.get(), condStack));

    auto ifThen = MakeIfStmt(MakeSymbolExpr("cond_var"), MakeSeqStmts({}));
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, ifThen, dynFixture.dynFunc.get(), condStack));

    EXPECT_NO_THROW(FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, MakeTensorOpStmt("CALL"),
                                       dynFixture.dynFunc.get(), condStack));
    EXPECT_NO_THROW(FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker,
                                       MakeTensorOpStmt("CALL", {{"callee", std::string("nonexistent")}}),
                                       dynFixture.dynFunc.get(), condStack));

    auto nested = MakeSeqStmts({MakeSeqStmts({})});
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, nested, dynFixture.dynFunc.get(), condStack));

    auto ifInBody = MakeIfStmt(MakeSymbolExpr("cond_var"), MakeSeqStmts({}));
    auto forWithIf = MakeForStmt(loopVar, MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1),
                                 MakeSeqStmts({ifInBody}));
    EXPECT_NO_THROW(
        FindExprFromIRStmt(irBackendCtx, cache, *linkerFixture.linker, forWithIf, dynFixture.dynFunc.get(), condStack));
}

TEST_F(TestSuite_IrBackend, FindAllExpressionFromIR_AllCases)
{
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        FunctionCache cache;
        dynFixture.dynFunc->body_ = std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{}, Sp());
        EXPECT_NO_THROW(FindAllExpressionFromIR(irBackendCtx, cache, *linkerFixture.linker, dynFixture.dynFunc.get()));
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        FunctionCache cache;
        dynFixture.dynFunc->body_ = nullptr;
        EXPECT_NO_THROW(FindAllExpressionFromIR(irBackendCtx, cache, *linkerFixture.linker, dynFixture.dynFunc.get()));
    }
    {
        IrBackendContext irBackendCtx;
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        FunctionCache cache;
        dynFixture.dynFunc->SetFunctionType(FunctionType::STATIC);
        EXPECT_NO_THROW(FindAllExpressionFromIR(irBackendCtx, cache, *linkerFixture.linker, dynFixture.dynFunc.get()));
    }
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_NullAndEmpty)
{
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        EXPECT_NO_THROW(VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", nullptr,
                                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group,
                                                  ctx.rootTileDict, ctx.controlFlowOss, ctx.expressionOss,
                                                  ctx.exprHeaderOss, 0, "expr", ctx.exprSrcFiles, ctx.meta));
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        EXPECT_NO_THROW(VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto",
                                                  MakeSeqStmts({}), dynFixture.dynFunc.get(), ctx.slotIdxMapping,
                                                  ctx.group, ctx.rootTileDict, ctx.controlFlowOss, ctx.expressionOss,
                                                  ctx.exprHeaderOss, 0, "expr", ctx.exprSrcFiles, ctx.meta));
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", MakeTensorOpStmt("ADD"),
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        EXPECT_TRUE(ctx.controlFlowOss.str().empty());
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto stmt = MakeTensorOpStmt("CALL", {{"callee", std::string("nonexistent")}});
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", stmt,
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        EXPECT_TRUE(ctx.controlFlowOss.str().empty());
    }
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_IfStmt)
{
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto ifStmt = MakeIfStmt(MakeSymbolExpr("cond_var"), MakeSeqStmts({}), MakeSeqStmts({}));
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("if (") != std::string::npos);
        EXPECT_TRUE(output.find("} else {") != std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto ifStmt = MakeIfStmt(MakeSymbolExpr("cond_var"), MakeSeqStmts({}));
        VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", ifStmt,
                                  dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                  ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                                  ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("if (") != std::string::npos);
        EXPECT_TRUE(output.find("} else {") == std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto ifStmt = MakeIfStmt(MakeSymbolExpr("c"), MakeSeqStmts({}), MakeSeqStmts({}));
        EXPECT_NO_THROW(VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto",
                                                  MakeSeqStmts({ifStmt}), dynFixture.dynFunc.get(), ctx.slotIdxMapping,
                                                  ctx.group, ctx.rootTileDict, ctx.controlFlowOss, ctx.expressionOss,
                                                  ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles, ctx.meta));
        EXPECT_FALSE(ctx.controlFlowOss.str().empty());
    }
}

TEST_F(TestSuite_IrBackend, VisitIRStmtForControlFlow_ForStmt)
{
    DynFuncFixture dynFixture;
    LinkerFixture linkerFixture;
    ControlFlowCtx ctx;
    auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64), Sp());
    auto forStmt = MakeForStmt(loopVar, MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1),
                               MakeSeqStmts({}));
    VisitIRStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                              dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                              ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr", ctx.exprSrcFiles,
                              ctx.meta);
    auto output = ctx.controlFlowOss.str();
    EXPECT_TRUE(output.find("LOOP(") != std::string::npos);
    EXPECT_TRUE(output.find("VAR_loop_idx") != std::string::npos);
    EXPECT_TRUE(output.find("VALUE_loop_idx") != std::string::npos);
}

TEST_F(TestSuite_IrBackend, VisitForStmtForControlFlow_AllCases)
{
    auto mkFor = [](const std::vector<std::pair<std::string, std::any>>& attrs = {}) {
        auto loopVar = IRContext::Get().MakeVar("loop_idx", std::make_shared<ir::ScalarType>(ir::DataType::INT64),
                                                Sp());
        return AsForStmt(MakeForStmt(loopVar, MakeImmediateExpr(0), MakeImmediateExpr(10), MakeImmediateExpr(1),
                                     MakeSeqStmts({}), attrs));
    };
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto forStmt = mkFor({{"submit_before_loop", true}});
        VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                                   dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                   ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr",
                                   ctx.exprSrcFiles, ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("RUNTIME_FUNCKEY_LOOP_BARRIER") != std::string::npos);
        EXPECT_TRUE(output.find("LOOP(") != std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        config::SetRuntimeOption<int64_t>(DEVICE_SCHED_PARALLELISM, 8);
        auto forStmt = mkFor({{"parallel", true}});
        VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                                   dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                   ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr",
                                   ctx.exprSrcFiles, ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN") != std::string::npos);
        EXPECT_TRUE(output.find("RUNTIME_FUNCKEY_PARALLEL_FOR_END") != std::string::npos);
    }
    {
        DynFuncFixture dynFixture;
        LinkerFixture linkerFixture;
        ControlFlowCtx ctx;
        auto forStmt = mkFor();
        VisitForStmtForControlFlow(ctx.irBackendCtx, ctx.cache, *linkerFixture.linker, ".pypto", forStmt,
                                   dynFixture.dynFunc.get(), ctx.slotIdxMapping, ctx.group, ctx.rootTileDict,
                                   ctx.controlFlowOss, ctx.expressionOss, ctx.exprHeaderOss, 1, "expr",
                                   ctx.exprSrcFiles, ctx.meta);
        auto output = ctx.controlFlowOss.str();
        EXPECT_TRUE(output.find("PARALLEL_FOR_BEGIN") == std::string::npos);
        EXPECT_TRUE(output.find("PARALLEL_FOR_END") == std::string::npos);
    }
}
