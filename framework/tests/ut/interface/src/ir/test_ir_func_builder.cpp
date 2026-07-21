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
 * \file test_ir_func_builder.cpp
 * \brief Unit tests for RootFunctionBuilder::BuildPathFuncSlotScope via pass::CreateRootFunctions.
 *
 * Focuses on constructAssembleSlotList correctness:
 *   - Deduplication of assemble ops targeting the same intermediate tensor
 *   - Mixed scenario: function param excluded, intermediate tensor kept
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/tilefwk.h"

using namespace npu::tile_fwk;

namespace {
static constexpr int64_t TILE = 16;

ir::Span Sp() { return ir::Span("test_ir_func_builder", 1, 1); }

struct IrFuncSetup {
    npu::tile_fwk::IRBuilder builder;
    std::shared_ptr<npu::tile_fwk::Function> fwkFunc;
    LogicalTensors params;
    std::vector<ir::StmtPtr> stmts;

    IrFuncSetup(const std::string& name)
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

    LogicalTensorPtr MakeLocal(const std::string& name)
    {
        return builder.CreateTensorVar(*fwkFunc, DT_FP32, {TILE, TILE}, TileOpFormat::TILEOP_ND, name);
    }

    Operation& AddDassemble(const LogicalTensorPtr& src, const LogicalTensorPtr& dst)
    {
        auto& op = fwkFunc->AddRawOperation(Opcode::OP_ASSEMBLE, {src}, {dst}, Sp());
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(Offset{TILE, TILE}));
        op.SetAttribute("dassemble", true);
        stmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
        return op;
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

    // Wrap current stmts as the body of a ForStmt and push the ForStmt onto stmts.
    // ForStmt requires a non-null _config_scope attr (ConfigManagerNg::PushScope asserts non-null).
    ir::ForStmtPtr WrapStmtsInForLoop(const std::string& loopVarName,
                                      std::vector<std::pair<std::string, std::any>> attrs = {})
    {
        auto body = std::make_shared<ir::SeqStmts>(stmts, Sp());
        stmts.clear();
        auto intType = std::make_shared<ir::ScalarType>(ir::DataType::INT64);
        auto loopVar = IRContext::Get().MakeVar(loopVarName, intType, Sp());
        auto zero = std::make_shared<ir::ConstInt>(0, ir::DataType::INT64, Sp());
        auto ten = std::make_shared<ir::ConstInt>(10, ir::DataType::INT64, Sp());
        auto one = std::make_shared<ir::ConstInt>(1, ir::DataType::INT64, Sp());
        attrs.emplace_back("_config_scope", ConfigManagerNg::CurrentScope());
        auto forStmt = std::make_shared<ir::ForStmt>(loopVar, zero, ten, one, std::vector<ir::IterArgPtr>{}, body,
                                                     std::vector<ir::VarPtr>{}, Sp(), std::move(attrs));
        stmts.push_back(forStmt);
        return forStmt;
    }
};

std::vector<npu::tile_fwk::Function*> FindHiddenFuncs()
{
    std::vector<npu::tile_fwk::Function*> result;
    for (auto& [name, func] : Program::GetInstance().GetFunctionMap()) {
        if (name.find("_hiddenfunc") != std::string::npos) {
            result.push_back(func.get());
        }
    }
    return result;
}

std::vector<int> CollectConstructAssembleSlots()
{
    std::vector<int> slots;
    for (auto* func : FindHiddenFuncs()) {
        auto scope = func->GetSlotScope();
        if (scope) {
            for (int s : scope->constructAssembleSlotList) {
                slots.push_back(s);
            }
        }
    }
    return slots;
}
} // namespace

class IrFuncBuilderTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetBuildStatic(false);
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
    }
};

// ============================================================================
// Dedup: two dassemble ops targeting the same intermediate tensor
//        => constructAssembleSlotList should contain exactly 1 slot
// ============================================================================
TEST_F(IrFuncBuilderTest, TestConstructAssembleSlotList_DedupSameSlot)
{
    IrFuncSetup setup("DedupSameSlot");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");

    setup.AddDassemble(a, aux);
    setup.AddDassemble(a, aux);

    auto irFunc = setup.BuildIrFunction("DedupSameSlot");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);

    auto slots = CollectConstructAssembleSlots();
    EXPECT_EQ(slots.size(), 1u) << "Expected 1 slot (deduplicated), got " << slots.size();
}

// ============================================================================
// Mixed: one dassemble to function param (excluded) + one to intermediate (kept)
//        => constructAssembleSlotList should contain exactly 1 slot
// ============================================================================
TEST_F(IrFuncBuilderTest, TestConstructAssembleSlotList_MixedParamAndIntermediate)
{
    IrFuncSetup setup("MixedParamAndIntermediate");

    auto a = setup.MakeParam("a");
    auto out = setup.MakeParam("out");
    auto aux = setup.MakeLocal("aux");

    setup.AddDassemble(a, aux);
    setup.AddDassemble(aux, out);

    auto irFunc = setup.BuildIrFunction("MixedParamAndIntermediate");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);

    auto slots = CollectConstructAssembleSlots();
    EXPECT_EQ(slots.size(), 1u) << "Expected 1 slot (aux only, out excluded), got " << slots.size();
}

// ============================================================================
// ForStmt with "unroll_times" attr => TransformStmts reads it via std::any_cast<int>
//   and appends "_Unroll<N>" to the loop-var-derived path suffix baked into the
//   hidden func raw name (CreateHiddenFunc: dynFuncRaw + "_" + loopVarName + "_PATH0_hiddenfunc").
//   Verifies the unroll_times branch (ir_func_builder.cpp:521-527).
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_UnrollTimesAttr)
{
    IrFuncSetup setup("UnrollTimesAttr");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");
    setup.AddDassemble(a, aux);

    setup.WrapStmtsInForLoop("i", {{"unroll_times", 4}});

    auto irFunc = setup.BuildIrFunction("UnrollTimesAttr");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);
    EXPECT_NE(hiddenFuncs[0]->GetRawName().find("_Unroll4_"), std::string::npos)
        << "Expected '_Unroll4_' in hidden func raw name, got: " << hiddenFuncs[0]->GetRawName();
}

// ============================================================================
// ForStmt WITHOUT "unroll_times" attr => default unrollTimes=1, suffix "_Unroll1".
//   Baseline confirming the attr-absent path defaults to 1.
// ============================================================================
TEST_F(IrFuncBuilderTest, TestTransformStmts_UnrollTimesDefault)
{
    IrFuncSetup setup("UnrollTimesDefault");

    auto a = setup.MakeParam("a");
    auto aux = setup.MakeLocal("aux");
    setup.AddDassemble(a, aux);

    setup.WrapStmtsInForLoop("i");

    auto irFunc = setup.BuildIrFunction("UnrollTimesDefault");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    auto hiddenFuncs = FindHiddenFuncs();
    ASSERT_EQ(hiddenFuncs.size(), 1u);
    EXPECT_NE(hiddenFuncs[0]->GetRawName().find("_Unroll1_"), std::string::npos)
        << "Expected '_Unroll1_' in hidden func raw name, got: " << hiddenFuncs[0]->GetRawName();
}

TEST_F(IrFuncBuilderTest, TestMigrateReshapeInplaceLinkToHiddenFunc)
{
    IrFuncSetup setup("MigrateReshapeInplaceLink");

    auto input = setup.MakeParam("input");
    auto output = setup.MakeParam("output");
    auto reshaped = setup.MakeLocal("reshaped");

    auto& reshape = setup.fwkFunc->AddRawOperation(Opcode::OP_RESHAPE, {input}, {reshaped}, Sp());
    reshape.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    setup.fwkFunc->SetSameMemId(input, reshaped);
    setup.stmts.push_back(std::static_pointer_cast<const ir::Stmt>(reshape.shared_from_this()));

    auto reshapeStmts = setup.stmts;
    setup.stmts.clear();
    setup.AddDassemble(reshaped, output);
    setup.WrapStmtsInForLoop("i");
    setup.stmts.insert(setup.stmts.begin(), reshapeStmts.begin(), reshapeStmts.end());

    auto irFunc = setup.BuildIrFunction("MigrateReshapeInplaceLink");
    auto irProg = std::make_shared<ir::Program>(std::vector<ir::FunctionPtr>{irFunc}, "test", Sp());

    auto createRoot = pypto::ir::pass::CreateRootFunctions();
    (void)createRoot(irProg);

    Function* reshapeHiddenFunc = nullptr;
    for (auto* func : FindHiddenFuncs()) {
        for (auto& op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_RESHAPE) {
                reshapeHiddenFunc = func;
                break;
            }
        }
    }
    ASSERT_NE(reshapeHiddenFunc, nullptr);
    ASSERT_EQ(reshapeHiddenFunc->GetIncast().size(), 1u);
    ASSERT_EQ(reshapeHiddenFunc->GetOutcast().size(), 1u);

    auto incastRaw = reshapeHiddenFunc->GetIncast().front()->GetRawTensor();
    auto outcastRaw = reshapeHiddenFunc->GetOutcast().front()->GetRawTensor();
    EXPECT_EQ(outcastRaw->memoryId, incastRaw->memoryId);

    auto link = reshapeHiddenFunc->outIncastLinkMap.find(outcastRaw);
    ASSERT_NE(link, reshapeHiddenFunc->outIncastLinkMap.end());
    EXPECT_EQ(link->second, incastRaw);
}
