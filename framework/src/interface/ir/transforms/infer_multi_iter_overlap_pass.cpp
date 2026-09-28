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
 * \file infer_multi_iter_overlap_pass.cpp
 * \brief After create_root: Mark RebuildableMultiIterNoOverlap when each hidden-outcast raw
 *        has one hidden assemble writer whose writes are disjoint on every enclosing For
 *        (innermost → outer).
 *
 * Algorithm:
 *   0) ClearMarks on every Function in the IR program.
 *   1) Walk entry For→CALL→path→hidden: loop step/outer, creates, writers
 *      (only assemble dsts whose raw is in that hidden's GetOutcast()).
 *   2) Per rawMagic, innermost→outer For: create@layer stops as non-overlap;
 *      else SeparableUnderInduction(var, step, dyn toOffset, tile shape).
 *   3) All layers OK ⇒ Mark the writer hidden (copied to its root by SubgraphToFunction).
 *
 * Pre-cond: after create_root_functions; hidden assemble only; create via
 * path constructAssembleSlotList; dyn toOffset rank matches tile.
 */

#include "ir/transforms/passes.h"

#include "interface/configs/config_manager_ng.h"
#include "interface/function/function.h"
#include "interface/function/rebuildable_attribute.h"
#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/tensor/tensor_slot.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/stmt.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pypto::ir {
namespace {

using npu::tile_fwk::AssembleOpAttribute;
using npu::tile_fwk::CallOpAttribute;
using npu::tile_fwk::LogicalTensorPtr;
using npu::tile_fwk::Opcode;
using npu::tile_fwk::Operation;
using npu::tile_fwk::Program;
using npu::tile_fwk::RebuildableAttributeManager;
using npu::tile_fwk::RebuildableMultiIterNoOverlap;
using npu::tile_fwk::SymbolicScalar;
using FwkFunction = npu::tile_fwk::Function;

struct LoopInfo {
    VarPtr var;
    SymbolicScalar step;
    const Var* outer = nullptr; // nullptr for the outermost For
};

struct LoopWriter {
    Operation* op;
    const Var* loop; // innermost enclosing For
};

struct LoopContext {
    std::unordered_map<const Var*, LoopInfo> loopByVar;
    std::unordered_map<int, const Var*> createdAtLoopByRaw;
    std::unordered_map<int, std::vector<LoopWriter>> loopWritersByRaw;
};

FwkFunction& AsFrameworkFunction(const FunctionPtr& func)
{
    return const_cast<FwkFunction&>(*std::static_pointer_cast<const FwkFunction>(func));
}

bool IsAssembleLike(const Operation& op)
{
    return op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA;
}

bool ProvablyTrue(const SymbolicScalar& cond)
{
    SymbolicScalar simplified = cond.Simplify();
    return simplified.ConcreteValid() && simplified.Concrete() != 0;
}

// A var absent from offset gives offsetNext == offsetCur, so it is never separable.
bool SeparableUnderInduction(const LoopInfo& loop, const std::vector<SymbolicScalar>& offset,
                             const std::vector<SymbolicScalar>& shape)
{
    const SymbolicScalar sym = SymbolicScalar::FromExpr(loop.var);
    const std::unordered_map<VarPtr, ExprPtr> bumpMap{{loop.var, (sym + loop.step).AsExpr()}};
    for (size_t dim = 0; dim < offset.size(); ++dim) {
        SymbolicScalar offsetCur = offset[dim];
        SymbolicScalar offsetNext = offsetCur.SubstituteVars(bumpMap);
        if (ProvablyTrue(offsetNext - offsetCur >= shape[dim])) {
            return true;
        }
    }
    return false;
}

bool GetAssembleOffsetShape(const Operation& op, std::vector<SymbolicScalar>& offset,
                            std::vector<SymbolicScalar>& shape)
{
    auto attr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    offset = attr->GetToDynOffset();
    shape = SymbolicScalar::FromConcrete(op.GetIOperands()[0]->GetShape());
    return offset.size() == shape.size();
}

FwkFunction* ResolveCalleeFunction(const Operation& callOp)
{
    auto attr = std::static_pointer_cast<CallOpAttribute>(callOp.GetOpAttribute());
    return Program::GetInstance().GetFunctionByMagicName(attr->GetCalleeMagicName());
}

// Encode only queries root outcasts; non-outcast assemble dst lives in rootInner memory,
// so it is neither a writer nor a Mark candidate.
void RegisterLoopAssemble(Operation& op, const Var* loopVar, const std::unordered_set<int>& outcastRaws,
                          LoopContext& loops)
{
    for (const auto& o : op.GetOOperands()) {
        const int raw = o->GetRawMagic();
        if (outcastRaws.count(raw) != 0) {
            loops.loopWritersByRaw[raw].push_back({&op, loopVar});
        }
    }
}

// Create signal: path constructAssembleSlotList, whose slots are re-allocated on every path entry.
void RegisterCreates(FwkFunction& func, const Var* loopVar, LoopContext& loops)
{
    auto scope = func.GetSlotScope();
    if (scope == nullptr || scope->constructAssembleSlotList.empty()) {
        return;
    }
    // Inverse of FlushConstructAssembleSlots: list id == CreateTensor(*GetSlotTensor(lt)).GetId().
    auto slotMgr = Program::GetInstance().GetTensorSlotManager();
    std::unordered_set<int> slotIds(scope->constructAssembleSlotList.begin(), scope->constructAssembleSlotList.end());
    for (const auto& [lt, st] : slotMgr->slotTensorDict) {
        if (slotIds.count(static_cast<int>(st->Id())) != 0) {
            loops.createdAtLoopByRaw.emplace(lt->GetRawMagic(), loopVar);
        }
    }
}

void RegisterFromCallee(FwkFunction& func, const Var* loopVar, LoopContext& loops)
{
    if (func.IsHiddenFunction()) {
        std::unordered_set<int> outcastRaws;
        for (const auto& o : func.GetOutcast()) {
            outcastRaws.insert(o->GetRawMagic());
        }
        for (auto& op : func.Operations(false)) {
            if (IsAssembleLike(op)) {
                RegisterLoopAssemble(op, loopVar, outcastRaws, loops);
            }
        }
        return;
    }
    RegisterCreates(func, loopVar, loops);
    for (auto& op : func.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        if (auto* callee = ResolveCalleeFunction(op)) {
            RegisterFromCallee(*callee, loopVar, loops);
        }
    }
}

// Walk entry body: record each For's step + outer link; under a loop, follow OP_CALL to
// register creates (path constructAssembleSlotList) and hidden-outcast assemble writers.
void CollectLoopsAndWriters(const StmtPtr& stmt, const Var* loopVar, LoopContext& loops)
{
    switch (stmt->GetKind()) {
        case ObjectKind::ForStmt: {
            auto forStmt = As<ForStmt>(stmt);
            const Var* forVar = forStmt->loopVar_.get();
            loops.loopByVar[forVar] = LoopInfo{forStmt->loopVar_, SymbolicScalar::FromExpr(forStmt->step_), loopVar};
            CollectLoopsAndWriters(forStmt->body_, forVar, loops);
            break;
        }
        case ObjectKind::SeqStmts: {
            for (const auto& s : As<SeqStmts>(stmt)->stmts_) {
                CollectLoopsAndWriters(s, loopVar, loops);
            }
            break;
        }
        case ObjectKind::IfStmt: {
            auto ifStmt = As<IfStmt>(stmt);
            CollectLoopsAndWriters(ifStmt->thenBody_, loopVar, loops);
            if (ifStmt->elseBody_.has_value()) {
                CollectLoopsAndWriters(ifStmt->elseBody_.value(), loopVar, loops);
            }
            break;
        }
        case ObjectKind::TensorOpStmt: {
            if (loopVar == nullptr) {
                break;
            }
            auto* op = static_cast<Operation*>(const_cast<Stmt*>(stmt.get()));
            if (op->GetOpcode() != Opcode::OP_CALL) {
                break;
            }
            if (auto* callee = ResolveCalleeFunction(*op)) {
                RegisterFromCallee(*callee, loopVar, loops);
            }
            break;
        }
        default:
            break;
    }
}

bool TensorHasAtomicAdd(const LogicalTensorPtr& tensor)
{
    for (const auto& prod : tensor->GetProducers()) {
        if (prod->GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            return true;
        }
    }
    return false;
}

bool IsComputeDeterminismEnabled()
{
    using npu::tile_fwk::ConfigManagerNg;
    return ConfigManagerNg::GetGlobalConfig<int64_t>("compute_determinism_level") >= 1;
}

// Root functions do not exist yet; SubgraphToFunction::InitializeRootFunction copies these marks.
void MarkNoOverlap(FwkFunction& writer, int rawMagic)
{
    RebuildableAttributeManager::GetInstance().GetAttr<RebuildableMultiIterNoOverlap>(&writer)->Mark(rawMagic);
}

void InferRawMultiIterOverlap(int rawMagic, const std::vector<LoopWriter>& writers, const LoopContext& loops)
{
    if (writers.size() != 1) {
        return;
    }
    Operation* op = writers[0].op;
    if (IsComputeDeterminismEnabled() && TensorHasAtomicAdd(op->GetOOperands()[0])) {
        return;
    }

    std::vector<SymbolicScalar> offset;
    std::vector<SymbolicScalar> shape;
    if (!GetAssembleOffsetShape(*op, offset, shape)) {
        return;
    }

    const Var* createdAt = nullptr;
    auto createIt = loops.createdAtLoopByRaw.find(rawMagic);
    if (createIt != loops.createdAtLoopByRaw.end()) {
        createdAt = createIt->second;
    }

    // Inner → outer: creation at this layer ⇒ non-overlap + stop; else require separable.
    for (const Var* var = writers[0].loop; var != nullptr;) {
        if (createdAt == var) {
            break;
        }
        const LoopInfo& loop = loops.loopByVar.at(var);
        if (!SeparableUnderInduction(loop, offset, shape)) {
            return;
        }
        var = loop.outer;
    }
    MarkNoOverlap(*op->BelongTo(), rawMagic);
}

void ClearMultiIterMarks(const ProgramPtr& irProgram)
{
    auto& mgr = RebuildableAttributeManager::GetInstance();
    for (const auto& [name, funcPtr] : irProgram->functions_) {
        (void)name;
        mgr.GetAttr<RebuildableMultiIterNoOverlap>(&AsFrameworkFunction(funcPtr))->ClearMarks();
    }
}

} // namespace

Pass pass::InferMultiIterOverlap()
{
    return pass::CreateProgramPass(
        [](const ProgramPtr& irProgram) -> ProgramPtr {
            ClearMultiIterMarks(irProgram);
            LoopContext loops;
            for (const auto& [name, funcPtr] : irProgram->functions_) {
                (void)name;
                FwkFunction& f = AsFrameworkFunction(funcPtr);
                if (f.entry_ && funcPtr->body_ != nullptr) {
                    CollectLoopsAndWriters(funcPtr->body_, nullptr, loops);
                }
            }
            for (const auto& [rawMagic, writers] : loops.loopWritersByRaw) {
                InferRawMultiIterOverlap(rawMagic, writers, loops);
            }
            return irProgram;
        },
        "InferMultiIterOverlap");
}

} // namespace pypto::ir
