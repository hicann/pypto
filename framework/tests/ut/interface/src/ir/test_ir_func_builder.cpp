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

ir::Span Sp()
{
    return ir::Span("test_ir_func_builder", 1, 1);
}

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
    EXPECT_EQ(slots.size(), 1u)
        << "Expected 1 slot (deduplicated), got " << slots.size();
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
    EXPECT_EQ(slots.size(), 1u)
        << "Expected 1 slot (aux only, out excluded), got " << slots.size();
}
